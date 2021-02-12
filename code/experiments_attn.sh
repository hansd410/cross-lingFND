#python train.py --src_ref_dir ../data/NAIVE/Eng/ --tgt_ref_dir ../data/NAIVE/Kor/ --enRef_X_train en_train_data.txt --enRef_X_test en_test_data.txt --koRef_X_file ko_data.txt
#mv save/adan /mnt/hansd410/adanModel/noNERver1

#python train.py --attn avg
#mv save/adan /mnt/hansd410/adanModel/attAvg1
#
#python train.py --attn avg
#mv save/adan /mnt/hansd410/adanModel/attAvg2
#
#python train.py --attn avg
#mv save/adan /mnt/hansd410/adanModel/attAvg3
#
#python train.py --attn avg
#mv save/adan /mnt/hansd410/adanModel/attAvg4
#
#python train.py --attn avg
#mv save/adan /mnt/hansd410/adanModel/attAvg5
#
#
#python train.py --attn last
#mv save/adan /mnt/hansd410/adanModel/attLast1
#
#python train.py --attn last
#mv save/adan /mnt/hansd410/adanModel/attLast2
#
#python train.py --attn last
#mv save/adan /mnt/hansd410/adanModel/attLast3
#
#python train.py --attn last
#mv save/adan /mnt/hansd410/adanModel/attLast4
#
#python train.py --attn last
#mv save/adan /mnt/hansd410/adanModel/attLast5


python train.py --attn first
mv save/adan /mnt/hansd410/adanModel/attFirst1

#python train.py --attn first
#mv save/adan /mnt/hansd410/adanModel/attFirst2
#
#python train.py --attn first
#mv save/adan /mnt/hansd410/adanModel/attFirst3
#
#python train.py --attn first
#mv save/adan /mnt/hansd410/adanModel/attFirst4
#
#python train.py --attn first
#mv save/adan /mnt/hansd410/adanModel/attFirst5
#
#
#python train.py --attn dot
#mv save/adan /mnt/hansd410/adanModel/attDot1
#
#python train.py --attn dot
#mv save/adan /mnt/hansd410/adanModel/attDot2
#
#python train.py --attn dot
#mv save/adan /mnt/hansd410/adanModel/attDot3
#
#python train.py --attn dot
#mv save/adan /mnt/hansd410/adanModel/attDot4
#
#python train.py --attn dot
#mv save/adan /mnt/hansd410/adanModel/attDot5

# ref option
# --ref False

# bwe option
# --emb_filename ../data/embedding/BWE/koEnBWE.txt

# bne option
# --refEmb_filename ../data/embedding/BNE/koEnBNE.txt

# ner shuffle option
# --src_ref_dir ../data/fakeNews/adanData/NER/shuffle/ --enRef_X_train shuffle_personOrg_ner_train.txt --enRef_Y_train trainLabel.txt --enRef_X_test shuffle_personOrg_ner_test.txt --enRef_Y_test testLabel.txt
# --tgt_ref_dir ../data/fakeNews/koreanData/nerData/shuffle/ --koRef_X_file shuffle_personOrg_ner.txt --koRef_Y_file label.txt

# ner person option
# --src_ref_dir ../data/fakeNews/adanData/NER/split/ --enRef_X_train person_ner_train.txt --enRef_Y_train trainLabel.txt --enRef_X_test person_ner_test.txt --enRef_Y_test testLabel.txt
# --tgt_ref_dir ../data/fakeNews/koreanData/nerData/ --koRef_X_file person_ner.txt --koRef_Y_file label.txt

# ner org option
# --src_ref_dir ../data/fakeNews/adanData/NER/split/ --enRef_X_train org_ner_train.txt --enRef_Y_train trainLabel.txt --enRef_X_test org_ner_test.txt --enRef_Y_test testLabel.txt
# --tgt_ref_dir ../data/fakeNews/koreanData/nerData/ --koRef_X_file org_ner.txt --koRef_Y_file label.txt

# dan option
# --model lstm
