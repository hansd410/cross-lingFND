import os

listDir = os.listdir('.')

fout = open("wholeResult.txt",'w')


for dirName in listDir:
	if('txt' in dirName):
		continue
	if('py' in dirName):
		continue
	fin = open(dirName+"/log.txt",'r')


	maxScore = 0
	line9 = ""
	line8 = ""
	line7 = ""
	line6 = ""
	line5 = ""
	line4 = ""
	line3 = ""
	line2 = ""
	line1 = ""
	while True:
		line = fin.readline()
		if not line : break
		if("Accuracy on 1803 samples" in line):
			score = float(line5.split(' ')[-1].rstrip('%\n'))
#			print(score)
			if(maxScore<score):
				maxScore = score
				enF1 = float(line8.split(' ')[-1].rstrip('%\n'))
				enPrec = float(line7.split(' ')[-1].rstrip('%\n'))
				enRecall = float(line6.split(' ')[-1].rstrip('%\n'))
				enAcc = maxScore
				koF1 = float(line3.split(' ')[-1].rstrip('%\n'))
				koPrec =float(line2.split(' ')[-1].rstrip('%\n'))
				koRecall = float(line1.split(' ')[-1].rstrip('%\n'))
				koAcc = float(line.split(' ')[-1].rstrip('%\n'))

		line9 = line8
		line8 = line7
		line7 = line6
		line6 = line5
		line5 = line4
		line4 = line3
		line3 = line2
		line2 = line1
		line1 = line

	fout.write(dirName+'\t'+str(koAcc)+'\t'+str(koF1)+'\t'+str(koPrec)+'\t'+str(koRecall)+'\n')

	

