import random
import os 
import sys

in_file = sys.argv[1] # *.narrowPeak
name = in_file.split('/')[1].split('.')[0]
file = open('maize4.chrmsize').readlines()[:19]
dic = { line.split()[0]:int(line.split()[1])-200 for line in file}

#print dic

#tar_file = open('rand_seq.txt','w')

if int(os.popen('wc -l %s' % in_file).read().split()[0]) > 3000:
#	recm = int(os.popen('wc -l %s' % in_file).read().split()[0])
	recm = 20020	
else:
	recm = 3000

n=0
while n < recm:
	chrm = random.choice(dic.keys())
	summit = random.randrange(dic[chrm])
	os.system('echo %s"\t"%s"\t"%s >> tmp.txt' % (chrm,summit-100,summit+101))
	n += 1

os.system('bedtools intersect -a tmp.txt -b %s -v > tmp2.txt' % in_file )

os.system('seqtk subseq zm.fa tmp2.txt | grep -v \'>\' | awk \'{print 0"\t"$0}\' >> train/%s_train.txt' % name)

os.system('rm tmp*.txt')
