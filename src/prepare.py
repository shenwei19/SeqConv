#import pandas as pd
import numpy as np
#import math
import os
import sys

in_file1 = sys.argv[1] # *.narrowPeak
in_file2 = sys.argv[2] # *.fa
out_file = sys.argv[3] # *train.txt

os.system('seqtk subseq zm.fa %s > fa/%s' % (in_file1,in_file2)) 
#file1 = open(in_file1).readlines()
file2 = open('fa/'+in_file2).readlines()

tar_file = open('train/'+out_file,'w')

for i in range(len(file2)/2):
	tar_file.write('1'+'\t'+file2[2*i+1])

tar_file.close()

#df = pd.DataFrame()

#for i,line in enumerate(file2):
	
#for line in file1:
#	l = line.split()
#	pat = l[0]+':'+str(int(l[1])+1)+'-'+l[2]
#	val = l[6]
#	for i,Line in enumerate(file2):
#		if Line[1:-1] == pat:
#			w = pd.DataFrame([1,file2[i+1][:-1]]).T
#			df = pd.concat([df,w])

#df[0] = math.log(pd.DataFrame(df[0])[0].astype('float64'),2)
#df2 = pd.DataFrame([1 for i in df[0].astype('float64')])
#for i in range(len(df)):
#	df.iloc[i,0] = df2.iloc[i,0]

#df1 = df[df[0]<df[0].quantile(0.99)]
#df1[0] = 1
#df1[0] = (df1[0]-min(df1[0]))/(max(df1[0])-min(df1[0]))

#df.to_csv(out_file,sep='\t',index=None,header=None)



