import numpy as np
import os
import sys

in_file1 = sys.argv[1] # *.narrowPeak
in_file2 = sys.argv[2] # *.fa
out_file = sys.argv[3] # *train.txt

os.system('seqtk subseq zm.fa %s > fa/%s' % (in_file1,in_file2)) 
file2 = open('fa/'+in_file2).readlines()

tar_file = open('train/'+out_file,'w')

for i in range(len(file2)/2):
	tar_file.write('1'+'\t'+file2[2*i+1])

tar_file.close()

