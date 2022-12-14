import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=30)
parser.add_argument('--ref', type=str, default='True')
parser.add_argument('--refOnly', type=str, default='False')
# parser.add_argument('--dataset', default='yelp')  # yelp, yelp-aren or amazon
# path to the datasets
parser.add_argument('--src_data_dir', default='../data/NAIVE/Eng/')
parser.add_argument('--tgt_data_dir', default='../data/NAIVE/Kor/')
parser.add_argument('--src_ref_dir', default='../data/NAIVE/Eng/reference/person')
parser.add_argument('--tgt_ref_dir', default='../data/NAIVE/Kor/reference/')

parser.add_argument('--yelp_X_train', default='en_train_data.txt')
parser.add_argument('--yelp_Y_train', default='en_train_label.txt')
parser.add_argument('--yelp_X_test', default='en_test_data.txt')
parser.add_argument('--yelp_Y_test', default='en_test_label.txt')

parser.add_argument('--enRef_X_train', default='person_ner_train.txt')
parser.add_argument('--enRef_Y_train', default='en_train_label.txt')
parser.add_argument('--enRef_X_test', default='person_ner_test.txt')
parser.add_argument('--enRef_Y_test', default='en_test_label.txt')

parser.add_argument('--chn_X_file', default='ko_data.txt')
parser.add_argument('--chn_Y_file', default='ko_label.txt')

parser.add_argument('--koRef_X_file', default='person_ner.txt')
parser.add_argument('--koRef_Y_file', default='ko_label.txt')

parser.add_argument('--en_train_lines', type=int, default=0)  # set to 0 to use all
parser.add_argument('--ch_train_lines', type=int, default=0)  # set to 0 to use all
parser.add_argument('--max_seq_len', type=int, default=200) # set to 0 to not truncate
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--model_save_file', default='./save/adan')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--Q_learning_rate', type=float, default=0.0005)
# path to BWE
parser.add_argument('--emb_filename', default='../data/embedding/BWE/koEnBWE.txt')
parser.add_argument('--refEmb_filename', default='../data/embedding/BNE/koEnBNE.txt')
parser.add_argument('--fix_emb', action='store_true')
parser.add_argument('--random_emb', action='store_true')
# use a fixed <unk> token for all words without pretrained embeddings when building vocab
parser.add_argument('--fix_unk', action='store_true')
parser.add_argument('--emb_size', type=int, default=100)
parser.add_argument('--model', default='cnn')  # dan or lstm or cnn
parser.add_argument('--f2model', default='lstm')  # dan or lstm or cnn
# for LSTM model
parser.add_argument('--attn', default='dot')  # attention mechanism (for LSTM): avg, last,first , dot
parser.add_argument('--bdrnn', dest='bdrnn', action='store_true', default=True)  # bi-directional LSTM
# use deep averaging network or deep summing network (for DAN model)
parser.add_argument('--sum_pooling/', dest='sum_pooling', action='store_true')
parser.add_argument('--avg_pooling/', dest='sum_pooling', action='store_false')
# for CNN model
parser.add_argument('--kernel_num', type=int, default=400)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,5])
parser.add_argument('--hidden_size', type=int, default=900)

parser.add_argument('--F_layers', type=int, default=1)
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--Q_layers', type=int, default=2)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--lambd', type=float, default=0.01)
parser.add_argument('--F_bn/', dest='F_bn', action='store_true')
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--P_bn/', dest='P_bn', action='store_true', default=True)
parser.add_argument('--no_P_bn/', dest='P_bn', action='store_false')
parser.add_argument('--Q_bn/', dest='Q_bn', action='store_true', default=True)
parser.add_argument('--no_Q_bn/', dest='Q_bn', action='store_false')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clip_lower', type=float, default=-0.01)
parser.add_argument('--clip_upper', type=float, default=0.01)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')
opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'
