import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import sys

train_file = sys.argv[1] # *_train.txt
name = sys.argv[2] # prefix

np.random.seed(12580)

def Seq2mat(seq):
	mat1 = np.zeros([4,len(seq)])
	mat2 = np.zeros([4,len(seq)])
	for letter,index in zip(seq,range(len(seq))):
		if letter == 'A' or letter == 'a':
			mat1[0,index] = 1
			mat2[3,len(seq)-index-1]=1
		if letter == 'C' or letter == 'c':
			mat1[1,index] = 1
                        mat2[2,len(seq)-index-1]=1
		if letter == 'G' or letter == 'g':
			mat1[2,index] = 1
                        mat2[1,len(seq)-index-1]=1
		if letter == 'T' or letter == 't':
			mat1[3,index] = 1
                        mat2[0,len(seq)-index-1]=1
		if letter == 'N':
			pass
	return mat1,mat2

def mat2Mat(filename):
	Mat = []
	Tag = []
	Seq = []
	with open(filename) as f:
		for line in f.readlines():
			tag = line.split()[0]
			seq = line.split()[1]
			if len(seq) == 201:
				Seq.append(seq)
				mat1,mat2 = Seq2mat(seq)
				Mat.append([mat1,mat2])
				Tag=np.append(Tag,float(tag))
	return np.array(Mat),Tag,Seq

os.system('head -n 500 %s > tmp.txt' % train_file)
Mat,Tag,Seq = mat2Mat('tmp.txt')
#Tag = Tag - 0.5

#Mat_1 = Mat[:,0,:,:]
#Mat_2 = Mat[:,1,:,:]
from keras.utils import np_utils
from collections import Counter

#x_train,x_test, y_train,y_test, seq_train,seq_test=train_test_split(Mat,Tag,Seq,test_size=1,random_state=12580)
#print seq_test
#print y_train
#print y_test

#x_train_1 = x_train[:,0,:,:]
#x_train_1 = x_train_1.reshape(x_train_1.shape[0],4,201,1)
#x_train_2 = x_train[:,1,:,:]
#x_train_2 = x_train_2.reshape(x_train_2.shape[0],4,201,1)

x_test_1 = Mat[:,0,:,:]
x_test_1 = x_test_1.reshape(x_test_1.shape[0],4,201,1)
x_test_2 = Mat[:,1,:,:]
x_test_2 = x_test_2.reshape(x_test_2.shape[0],4,201,1)

#y_train = np_utils.to_categorical(y_train,num_classes=len(Counter(Tag)))
#y_test = np_utils.to_categorical(y_test,num_classes=len(Counter(Tag)))


from keras.models import Model
from keras.models import load_model
from tqdm import tqdm
model_2 = load_model('%s_model.h5' % name)

#for layer in model_2.layers:
#	print layer.name

#from keras.models import Input

#x1 = Input(shape=(4,201,1))
#x2 = Input(shape=(4,201,1))

#can not use model_3 = Model(inputs=model_2.input,outputs=model_2.get_layer('conv2d_1').output), or report 'AttributeError: Layer conv2d_1 has multiple inbound nodes, hence the notion of "layer output" is ill-defined. Use `get_output_at(node_index)` instead' in case of various output shape even if we know the output shapes are identical

model_3 = Model(inputs=model_2.input,outputs=[model_2.get_layer('re_lu_1').get_output_at(0),model_2.get_layer('re_lu_2').get_output_at(0)]) # output a list with 2 elements 

def revComp(seq):
	rev_seq = ''
	for i in seq:
		if i == 'A' or i == 'a':
			rev_seq = 'T'+rev_seq
		if i == 'C' or i == 'c':
			rev_seq = 'G'+rev_seq
		if i == 'G' or i == 'g':
			rev_seq = 'C'+rev_seq
		if i == 'T' or i == 't':
			rev_seq = 'A'+rev_seq
		if i == 'N':	
			rev_seq = 'N'+rev_seq
	return rev_seq

seq = []
for i in tqdm(range(len(x_test_1))):
	dic = {}
	l0 = model_3.predict([x_test_1[i].reshape(1,4,201,1),x_test_2[i].reshape(1,4,201,1)])[0].reshape(178,16)
	l1 = model_3.predict([x_test_1[i].reshape(1,4,201,1),x_test_2[i].reshape(1,4,201,1)])[1].reshape(178,16)
	if max(max(l0.flatten()),max(l1.flatten())) > 0:
		for j in range(16):
			l0_index = np.where(l0[:,j]==max(l0[:,j]))[0][0] #np.where returns a tuple with one array
			l0_max = max(l0[:,j])
			l1_index = np.where(l1[:,j]==max(l1[:,j]))[0][0]
			l1_max = max(l1[:,j])
#			dict['+_'+str(l0_index)] = l0_max
#			dict['-_'+str(l1_index)] = l1_max
	#		print l0_index, l0_max, l1_index, l1_max
			if not '+_'+str(l0_index) in dic.keys():
				dic['+_'+str(l0_index)] = l0_max
			else:
				if dic['+_'+str(l0_index)] >= l0_max:
					pass
				else:
					dic['+_'+str(l0_index)] = l0_max
		
			if not '-_'+str(l1_index) in dic.keys():
	                        dic['-_'+str(l1_index)] = l1_max
	                else:
	                        if dic['-_'+str(l1_index)] >= l1_max:
	                                pass
	                        else:
	                                dic['-_'+str(l1_index)] = l1_max
		strand,seq_index = max(dic,key=dic.get).split('_') #this is the method to find max_num in a dict
	#	print type(seq_index) numpy.int64
	#	print seq_test[i]
		if strand == '+':
			seq_24 = Seq[i][int(seq_index):int(seq_index)+24]
		if strand == '-':	
			seq_24 = revComp(Seq[i])[int(seq_index):int(seq_index)+24]
		seq.append(seq_24)

with open('seq.txt','w') as tar_file:
	for i in range(len(seq)):
		tar_file.write(seq[i]+'\n')

os.system('rm tmp.txt')

from collections import Counter
#import tqdm
file=open('seq.txt').readlines()
tar_file=open('%s_seq.meme' % name,'w')
#tar_file.write('MEME version 4.4 \nALPHABET= ACGT \nstrands: + - \nBackground letter frequencies (from uniform background): \nA 0.25000 C 0.25000 G 0.25000 T 0.25000 \nMOTIF 0\nletter-probability matrix: alength= 4 w= 8 nsites= 445 E= 2.2e-019\n')

length=24
for i in range(length):
        list=[]
        for j in file:
                list.append(j[i])
#	print len(list)
        c = dict(Counter(list))
        sum_all = float(sum(c.values()))
        if 'A' in c.keys():
                freq_A = c['A']/sum_all
        else:
                freq_A = 0
        if 'C' in c.keys():
                freq_C = c['C']/sum_all
        else:
                freq_C = 0
        if 'G' in c.keys():
                freq_G = c['G']/sum_all
        else:
                freq_G = 0
        if 'T' in c.keys():
                freq_T = c['T']/sum_all
        else:
                freq_T = 0
        tar_file.write(str(freq_A)+'\t'+str(freq_C)+'\t'+str(freq_G)+'\t'+str(freq_T)+'\n')

tar_file.close()

os.system('rm seq.txt')
