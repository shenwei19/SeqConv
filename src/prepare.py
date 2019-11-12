import numpy as np
import os
import sys

ref_genome = sys.argv[1]
in_file1 = sys.argv[2] # *.narrowPeak
#in_file2 = sys.argv[3] # *.fa
out_file = sys.argv[3] # *train.txt
tmp_fa = 'tmp.fa'

os.system('seqtk subseq %s %s > %s' % (ref_genome,in_file1,tmp_fa)) 
file2 = open(tmp_fa).readlines()

tar_file = open(out_file,'w')

for i in range(len(file2)/2):
	tar_file.write('1'+'\t'+file2[2*i+1])

tar_file.close()
os.system('rm %s' % tmp_fa)
