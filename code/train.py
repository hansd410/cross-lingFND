# f2 put to q
# f2 dan option

import argparse
import logging
import os
import random
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

from data_prep.yelp_dataset import get_yelp_datasets
from data_prep.chn_hotel_dataset import get_chn_htl_datasets
from models import *
from options import opt
from vocab import Vocab
import utils

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)

# save logs
if not os.path.exists(opt.model_save_file):
	os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
 
# output options
log.info('Training ADAN with options:')
log.info(opt)
 

def train(opt):
	# vocab
	log.info(f'Loading Embeddings...')
	vocab = Vocab(opt.emb_filename)
	vocab2 = Vocab(opt.refEmb_filename)
	# datasets
	log.info(f'Loading data...')
#	yelp_X_train = os.path.join(opt.src_data_dir, 'X_train.txt.tok.shuf.lower')
#	yelp_Y_train = os.path.join(opt.src_data_dir, 'Y_train.txt.shuf')
#	yelp_X_test = os.path.join(opt.src_data_dir, 'X_test.txt.tok.lower')
#	yelp_Y_test = os.path.join(opt.src_data_dir, 'Y_test.txt')

	yelp_X_train = os.path.join(opt.src_data_dir, opt.yelp_X_train)
	yelp_Y_train = os.path.join(opt.src_data_dir, opt.yelp_Y_train)
	yelp_X_test = os.path.join(opt.src_data_dir, opt.yelp_X_test)
	yelp_Y_test = os.path.join(opt.src_data_dir, opt.yelp_Y_test)

	yelp_train, yelp_valid = get_yelp_datasets(vocab, yelp_X_train, yelp_Y_train,
			opt.en_train_lines, yelp_X_test, yelp_Y_test, opt.max_seq_len)

	# reference information
	enRef_X_train = os.path.join(opt.src_ref_dir, opt.enRef_X_train)
	enRef_Y_train = os.path.join(opt.src_ref_dir, opt.enRef_Y_train)
	enRef_X_test = os.path.join(opt.src_ref_dir, opt.enRef_X_test)
	enRef_Y_test = os.path.join(opt.src_ref_dir, opt.enRef_Y_test)

	enRef_train, enRef_valid = get_yelp_datasets(vocab2, enRef_X_train, enRef_Y_train,
			opt.en_train_lines, enRef_X_test, enRef_Y_test, opt.max_seq_len)

#	chn_X_file = os.path.join(opt.tgt_data_dir, 'X.sent.txt.shuf.lower')
#	chn_Y_file = os.path.join(opt.tgt_data_dir, 'Y.txt.shuf')

	chn_X_file = os.path.join(opt.tgt_data_dir, opt.chn_X_file)
	chn_Y_file = os.path.join(opt.tgt_data_dir, opt.chn_Y_file)

	chn_train, chn_valid, chn_test = get_chn_htl_datasets(vocab, chn_X_file, chn_Y_file,
			opt.ch_train_lines, opt.max_seq_len)

	# reference information
	koRef_X_file = os.path.join(opt.tgt_ref_dir, opt.koRef_X_file)
	koRef_Y_file = os.path.join(opt.tgt_ref_dir, opt.koRef_Y_file)

	koRef_train, koRef_valid, koRef_test = get_chn_htl_datasets(vocab2, koRef_X_file, koRef_Y_file,
			opt.ch_train_lines, opt.max_seq_len)

	log.info('Done loading datasets.')
	opt.num_labels = yelp_train.num_labels

	if opt.max_seq_len <= 0:
		# set to true max_seq_len in the datasets
		opt.max_seq_len = max(yelp_train.get_max_seq_len(),
							  chn_train.get_max_seq_len())


	# dataset loaders
	my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
	my_collate_f2 = utils.sorted_collate if opt.f2model=='lstm' else utils.unsorted_collate
	yelp_train_loader = DataLoader(yelp_train, opt.batch_size,
			shuffle=False, collate_fn=my_collate)
	yelp_train_loader_Q = DataLoader(yelp_train,
									 opt.batch_size,
									 shuffle=False, collate_fn=my_collate)
	chn_train_loader = DataLoader(chn_train, opt.batch_size,
			shuffle=False, collate_fn=my_collate)
	chn_train_loader_Q = DataLoader(chn_train,
									opt.batch_size,
									shuffle=False, collate_fn=my_collate)

	enRef_train_loader = DataLoader(enRef_train, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)
	enRef_train_loader_Q = DataLoader(enRef_train,
									 opt.batch_size,
									 shuffle=False, collate_fn=my_collate_f2)

	koRef_train_loader = DataLoader(koRef_train, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)
	koRef_train_loader_Q = DataLoader(koRef_train, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)

	# dataset iterators
	yelp_train_iter_Q = iter(yelp_train_loader_Q)
	enRef_train_iter_Q = iter(enRef_train_loader_Q)

	chn_train_iter = iter(chn_train_loader)
	chn_train_iter_Q = iter(chn_train_loader_Q)

	koRef_train_iter = iter(koRef_train_loader)
	koRef_train_iter_Q = iter(koRef_train_loader_Q)


	yelp_valid_loader = DataLoader(yelp_valid, opt.batch_size,
			shuffle=False, collate_fn=my_collate)
	chn_valid_loader = DataLoader(chn_valid, opt.batch_size,
			shuffle=False, collate_fn=my_collate)
	chn_test_loader = DataLoader(chn_test, opt.batch_size,
			shuffle=False, collate_fn=my_collate)

	enRef_valid_loader = DataLoader(enRef_valid, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)
	koRef_valid_loader = DataLoader(koRef_valid, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)
	koRef_test_loader = DataLoader(koRef_test, opt.batch_size,
			shuffle=False, collate_fn=my_collate_f2)

	# models
	if opt.model.lower() == 'dan':
		F = DANFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn)
	elif opt.model.lower() == 'lstm':
		F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
				opt.bdrnn, opt.attn)
	elif opt.model.lower() == 'cnn':
		F = CNNFeatureExtractor(vocab, opt.F_layers,
				opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
	else:
		raise Exception('Unknown model')

	if opt.f2model.lower() == 'dan':
		F2 = DANFeatureExtractor(vocab2, opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn)
	elif opt.f2model.lower() == 'lstm':
		F2 = LSTMFeatureExtractor(vocab2, opt.F_layers, opt.hidden_size, opt.dropout,
				opt.bdrnn, opt.attn)
	elif opt.f2model.lower() == 'cnn':
		F2 = CNNFeatureExtractor(vocab2, opt.F_layers,
				opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
	else:
		raise Exception('Unknown model')


	if(opt.ref == 'True'):
		P = SentimentClassifier(opt.P_layers, opt.hidden_size*2, opt.num_labels,
			opt.dropout, opt.P_bn)
		Q = LanguageDetector(opt.Q_layers, opt.hidden_size*2, opt.dropout, opt.Q_bn)
	else:
		P = SentimentClassifier(opt.P_layers, opt.hidden_size, opt.num_labels,
			opt.dropout, opt.P_bn)
		Q = LanguageDetector(opt.Q_layers, opt.hidden_size, opt.dropout, opt.Q_bn)
	F,F2, P, Q = F.to(opt.device),F2.to(opt.device), P.to(opt.device), Q.to(opt.device)
	optimizer = optim.Adam(list(F.parameters())+list(F2.parameters()) + list(P.parameters()),
						   lr=opt.learning_rate)
	optimizerQ = optim.Adam(Q.parameters(), lr=opt.Q_learning_rate)

	# training
	best_acc = 0.0
	best_f1 = 0.0
	best_f1_accuracy = 0.0
	best_f1_precision = 0.0
	best_f1_recall = 0.0
	for epoch in range(opt.max_epoch):
		F.train()
		F2.train()
		P.train()
		Q.train()
		yelp_train_iter = iter(yelp_train_loader)
		enRef_train_iter = iter(enRef_train_loader)

		# training accuracy
		correct, total = 0, 0
		sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
		grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
		for i, (inputs_en, targets_en) in tqdm(enumerate(yelp_train_iter),
											   total=len(yelp_train)//opt.batch_size):

			inputs_enRef, _ = next(enRef_train_iter)

			try:
				inputs_ch, _ = next(chn_train_iter)  # Chinese labels are not used
				inputs_koRef, _ = next(koRef_train_iter)
			except:
				# check if Chinese data is exhausted
				chn_train_iter = iter(chn_train_loader)
				inputs_ch, _ = next(chn_train_iter)

				koRef_train_iter = iter(koRef_train_loader)
				inputs_koRef, _ = next(koRef_train_iter)

			# Q iterations
			n_critic = opt.n_critic
			if n_critic>0 and ((epoch==0 and i<=25) or (i%500==0)):
				n_critic = 10
			utils.freeze_net(F)
			utils.freeze_net(F2)
			utils.freeze_net(P)
			utils.unfreeze_net(Q)
			for qiter in range(n_critic):
				# clip Q weights
				for p in Q.parameters():
					p.data.clamp_(opt.clip_lower, opt.clip_upper)
				Q.zero_grad()
				# get a minibatch of data
				try:
					# labels are not used
					q_inputs_en, _ = next(yelp_train_iter_Q)
					q_inputs_enRef, _ = next(enRef_train_iter_Q)
				except StopIteration:
					# check if dataloader is exhausted
					yelp_train_iter_Q = iter(yelp_train_loader_Q)
					q_inputs_en, _ = next(yelp_train_iter_Q)
					enRef_train_iter_Q = iter(enRef_train_loader_Q)
					q_inputs_enRef, _ = next(enRef_train_iter_Q)
				try:
					q_inputs_ch, _ = next(chn_train_iter_Q)
					q_inputs_koRef, _ = next(koRef_train_iter_Q)
				except StopIteration:
					chn_train_iter_Q = iter(chn_train_loader_Q)
					q_inputs_ch, _ = next(chn_train_iter_Q)
					koRef_train_iter_Q = iter(koRef_train_loader_Q)
					q_inputs_koRef, _ = next(koRef_train_iter_Q)

				features_en = F(q_inputs_en)

				if(opt.ref == 'True'):
					features_enRef = F2(q_inputs_enRef)
					o_en_ad = Q(torch.cat((features_en,features_enRef),dim=1))
				else:
					o_en_ad = Q(features_en)

#				o_en_ad = Q(features_en)
				l_en_ad = torch.mean(o_en_ad)
				(-l_en_ad).backward()
				log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
				sum_en_q = (sum_en_q[0] + 1, sum_en_q[1] + l_en_ad.item())

				features_ch = F(q_inputs_ch)
				if(opt.ref == 'True'):
					features_koRef = F2(q_inputs_koRef)
					o_ch_ad = Q(torch.cat((features_ch,features_koRef),dim=1))
				else:
					o_ch_ad = Q(features_ch)
				l_ch_ad = torch.mean(o_ch_ad)
				l_ch_ad.backward()
				log.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
				sum_ch_q = (sum_ch_q[0] + 1, sum_ch_q[1] + l_ch_ad.item())

				optimizerQ.step()

			# F&P iteration
			utils.unfreeze_net(F)
			utils.unfreeze_net(F2)
			utils.unfreeze_net(P)
			utils.freeze_net(Q)
			if opt.fix_emb:
				utils.freeze_net(F.word_emb)
				utils.freeze_net(F2.word_emb)
			# clip Q weights
			for p in Q.parameters():
				p.data.clamp_(opt.clip_lower, opt.clip_upper)
			F.zero_grad()
			F2.zero_grad()
			P.zero_grad()
			
			features_en = F(inputs_en)
			if(opt.ref == 'True'):
				features_enRef = F2(inputs_enRef)
				o_en_sent = P(torch.cat((features_en,features_enRef),dim=1))
			else:
				o_en_sent = P(features_en)
			l_en_sent = functional.nll_loss(o_en_sent, targets_en)
			l_en_sent.backward(retain_graph=True)


			if(opt.ref == 'True'):
				features_enRef = F2(inputs_enRef)
				o_en_ad = Q(torch.cat((features_en,features_enRef),dim=1))
			else:
				o_en_ad = Q(features_en)

			#o_en_ad = Q(features_en)
			l_en_ad = torch.mean(o_en_ad)
			(opt.lambd*l_en_ad).backward(retain_graph=True)
			# training accuracy
			_, pred = torch.max(o_en_sent, 1)
			total += targets_en.size(0)
			correct += (pred == targets_en).sum().item()

			features_ch = F(inputs_ch)
			if(opt.ref == 'True'):
				features_koRef = F2(inputs_koRef)
				o_ch_ad = Q(torch.cat((features_ch,features_koRef),dim=1))
			else:
				o_ch_ad = Q(features_ch)
			l_ch_ad = torch.mean(o_ch_ad)
			(-opt.lambd*l_ch_ad).backward()

			optimizer.step()
	
		# end of epoch
		log.info('Ending epoch {}'.format(epoch+1))
		# logs
		if sum_en_q[0] > 0:
			log.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
			log.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
		# evaluate
		log.info('Training Accuracy: {}%'.format(100.0*correct/total))
		log.info('Evaluating English Validation set:')
		evaluate(opt, yelp_valid_loader, enRef_valid_loader, F,F2, P)
		log.info('Evaluating Foreign validation set:')
		acc,f1,prec,recall = evaluate(opt, chn_valid_loader, koRef_valid_loader, F,F2, P)
		if f1 > best_f1:
			log.info(f'New Best Foreign f1: {f1}')
			best_f1 = f1 
			best_f1_accuracy = acc
			best_f1_precision = prec
			best_f1_recall = recall
			torch.save(F.state_dict(),
					'{}/netF_epoch_{}.pth'.format(opt.model_save_file, epoch))
			torch.save(F2.state_dict(),
					'{}/netF2_epoch_{}.pth'.format(opt.model_save_file, epoch))
			torch.save(P.state_dict(),
					'{}/netP_epoch_{}.pth'.format(opt.model_save_file, epoch))
			torch.save(Q.state_dict(),
					'{}/netQ_epoch_{}.pth'.format(opt.model_save_file, epoch))
		log.info('Evaluating Foreign test set:')
		evaluate(opt, chn_test_loader,koRef_test_loader, F,F2, P)
	
	log.info(f'Best Foreign f1: {best_f1}')
	log.info(f'Best Foreign f1 acc: {best_f1_accuracy}')
	log.info(f'Best Foreign f1 prec: {best_f1_precision}')
	log.info(f'Best Foreign f1 recall: {best_f1_recall}')


def evaluate(opt, loader, refLoader, F, F2, P):
	F.eval()
	F2.eval()
	P.eval()
	it = iter(loader)
	itRef = iter(refLoader)
	correct = 0
	total = 0
	predList = []
	targetList = []
	confusion = ConfusionMeter(opt.num_labels)
	with torch.no_grad():
		for inputs, targets in tqdm(it):
			refInputs, _ = next(itRef)
			if(opt.ref == 'True'):
				outputs = P(torch.cat((F(inputs),F2(refInputs)),dim=1))
			else:
				outputs = P(F(inputs))
			_, pred = torch.max(outputs, 1)
			confusion.add(pred.data, targets.data)
			total += targets.size(0)
			correct += (pred == targets).sum().item()
			predList.append(pred.data)
			targetList.append(targets.data)
	accuracy = correct / total

	predTensor = torch.cat(predList)
	targetTensor = torch.cat(targetList)
	# filp result for (fake = true class)
	predTensor = 1-predTensor
	targetTensor = 1-targetTensor
	predList = predTensor.tolist()
	targetList = targetTensor.tolist()
	f1Score = f1_score(targetList,predList,zero_division=1)
	precisionScore = precision_score(targetList,predList,zero_division=1)
	recallScore = recall_score(targetList,predList,zero_division=1)

	log.info('eval F1 {}'.format(f1Score))
	log.info('eval precision {}'.format(precisionScore))
	log.info('eval recall {}'.format(recallScore))
	log.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
	log.debug(confusion.conf)
	return accuracy, f1Score, precisionScore, recallScore


if __name__ == '__main__':
	train(opt)
