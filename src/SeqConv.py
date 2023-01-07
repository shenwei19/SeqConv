import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os 
import sys
from keras.utils import np_utils
from collections import Counter
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.layers import Convolution2D,MaxPooling2D,AveragePooling2D,ReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.layers import concatenate
from keras.initializers import random_normal
from keras import regularizers
from sklearn.metrics import roc_curve, auc, average_precision_score

np.random.seed(12580)

in_file = sys.argv[1]
name = in_file.split('_')[0]

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
	with open(filename) as f:
		for line in f.readlines():
			tag = line.split()[0]
			seq = line.split()[1]
			mat1,mat2 = Seq2mat(seq)
			Mat.append([mat1,mat2])
			Tag=np.append(Tag,float(tag))
	return np.array(Mat),Tag

Mat,Tag = mat2Mat('train/'+in_file)

x_train,x_test,y_train,y_test=train_test_split(Mat,Tag,test_size=0.1,random_state=12580)

x_train_1 = x_train[:,0,:,:]
x_train_1 = x_train_1.reshape(x_train_1.shape[0],4,201,1)
x_train_2 = x_train[:,1,:,:]
x_train_2 = x_train_2.reshape(x_train_2.shape[0],4,201,1)

x_test_1 = x_test[:,0,:,:]
x_test_1 = x_test_1.reshape(x_test_1.shape[0],4,201,1)
x_test_2 = x_test[:,1,:,:]
x_test_2 = x_test_2.reshape(x_test_2.shape[0],4,201,1)

mat_1 = Input(shape=(4,201,1))
mat_2 = Input(shape=(4,201,1))

shared_conv = Convolution2D(filters=16,kernel_size=(4,24),padding='valid') #no need to claim input_shape?

x1 = shared_conv(mat_1)
x2 = shared_conv(mat_2)

x1 = ReLU()(x1)
x2 = ReLU()(x2)

x1_1 = MaxPooling2D(pool_size=(1,178))(x1) #length-filter_length+1
x1_2 = AveragePooling2D(pool_size=(1,178))(x1)

x2_1 = MaxPooling2D(pool_size=(1,178))(x2)
x2_2 = AveragePooling2D(pool_size=(1,178))(x2)

merged_vector = concatenate([x1_1,x1_2,x2_1,x2_2])

x = Flatten()(merged_vector)
x = BatchNormalization()(x)
x = Dense(64,activation='relu',kernel_initializer=random_normal(mean=0,stddev=1),bias_initializer=random_normal(mean=0,stddev=1))(x)
main_output = Dense(1,activation='sigmoid',kernel_initializer=random_normal(mean=0,stddev=1),bias_initializer=random_normal(mean=0,stddev=1))(x)

model = Model(inputs=[mat_1,mat_2],outputs=main_output)
model.compile(loss='mse',optimizer='sgd')
early_stopping = EarlyStopping(monitor='val_loss',patience=100)

start = time.time()

history = model.fit([x_train_1,x_train_2],y_train,batch_size=32,epochs=1000,validation_data=[[x_test_1,x_test_2],y_test],callbacks=[early_stopping])

end = time.time()
dur = end - start
os.system('echo %s %s >> time_rec.txt' % (name,dur))

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(1,2,1)
plt.plot(range(1,len(loss)+1),loss,color='red')
plt.title('loss',fontsize=10)

plt.subplot(1,2,2)
plt.plot(range(1,len(val_loss)+1),val_loss,color='green')
plt.title('val_loss',fontsize=10)
plt.savefig("loss/%s_loss.png" % name)

loss_file = open('loss/%s_loss.txt' % name,'w')
loss_file.write('loss\tval_loss\n')
for i,j in zip(loss,val_loss):
	loss_file.write(str(i)+'\t'+str(j)+'\n')
loss_file.close()

prob = model.predict([x_test_1,x_test_2])
fpr, tpr, threshold = roc_curve(y_test,prob)
roc_auc = auc(fpr,tpr)
prc = average_precision_score(y_test,prob)
os.system('echo %s %s %s >> roc/%s_rec.txt' % (name,roc_auc,prc,name))

roc_file = open('roc/%s_roc.txt' % name,'w')
roc_file.write('fpr\ttpr\n')
for i,j in zip(fpr,tpr):
	roc_file.write(str(i)+'\t'+str(j)+'\n')
roc_file.close()

plt.figure()
plt.plot(fpr,tpr,label='area = %s' % roc_auc)
plt.title('ROC')
plt.savefig("roc/%s_roc.png" % name)

model.save('model/%s_model.h5' % name)
