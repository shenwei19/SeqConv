import random
import os 
import sys

chrm_file = sys.argv[1] # *.fa
in_file = sys.argv[2] # *.narrowPeak
train_file = sys.argv[3]
size_file = sys.argv[4] # *.size
recm = int(sys.argv[5])

file = open(size_file).readlines()
dic = { line.split()[0]:int(line.split()[1])-200 for line in file}

#if int(os.popen('wc -l %s' % in_file).read().split()[0]) > 6000:
#	recm = int(os.popen('wc -l %s' % in_file).read().split()[0])
#	recm = 20020	
#else:
#	recm = 4000

n=0
while n < recm:
	chrm = random.choice(dic.keys())
	summit = random.randrange(dic[chrm])
	os.system('echo %s"\t"%s"\t"%s >> tmp.txt' % (chrm,summit-100,summit+101))
	n += 1

os.system('bedtools intersect -a tmp.txt -b %s -v > tmp2.txt' % in_file )

os.system('seqtk subseq %s tmp2.txt | grep -v \'>\' | awk \'{print 0"\t"$0}\' >> %s' % (chrm_file,train_file))

os.system('rm tmp*.txt')
