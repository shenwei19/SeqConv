#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import os 
import sys

np.random.seed(12580)

in_file = sys.argv[1]
name = sys.argv[2]

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
			if len(seq) == 201:
				mat1,mat2 = Seq2mat(seq)
				Mat.append([mat1,mat2])
				Tag=np.append(Tag,float(tag))
	return np.array(Mat),Tag

Mat,Tag = mat2Mat(in_file)
#Tag = Tag - 0.5

#Mat_1 = Mat[:,0,:,:]
#Mat_2 = Mat[:,1,:,:]
from keras.utils import np_utils
from collections import Counter

x_train,x_test,y_train,y_test=train_test_split(Mat,Tag,test_size=0.1,random_state=12580)
#print y_train
#print y_test

x_train_1 = x_train[:,0,:,:]
x_train_1 = x_train_1.reshape(x_train_1.shape[0],4,201,1)
x_train_2 = x_train[:,1,:,:]
x_train_2 = x_train_2.reshape(x_train_2.shape[0],4,201,1)

x_test_1 = x_test[:,0,:,:]
x_test_1 = x_test_1.reshape(x_test_1.shape[0],4,201,1)
x_test_2 = x_test[:,1,:,:]
x_test_2 = x_test_2.reshape(x_test_2.shape[0],4,201,1)

#y_train = np_utils.to_categorical(y_train,num_classes=len(Counter(Tag)))
#y_test = np_utils.to_categorical(y_test,num_classes=len(Counter(Tag)))

#print x_train_1.shape,x_train_2.shape,x_test_1.shape,x_test_2.shape,y_train.shape,y_test.shape

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

mat_1 = Input(shape=(4,201,1))
mat_2 = Input(shape=(4,201,1))

shared_conv = Convolution2D(filters=16,kernel_size=(4,24),padding='valid') #no need to claim input_shape?
#shared_conv_2 = Convolution2D(filters=16,kernel_size=(4,24),padding='valid')

x1 = shared_conv(mat_1)
x2 = shared_conv(mat_2)

x1 = ReLU()(x1)
x2 = ReLU()(x2)

x1_1 = MaxPooling2D(pool_size=(1,178))(x1) #length-filter_length+1
x1_2 = AveragePooling2D(pool_size=(1,178))(x1)

x2_1 = MaxPooling2D(pool_size=(1,178))(x2)
x2_2 = AveragePooling2D(pool_size=(1,178))(x2)

merged_vector = concatenate([x1_1,x1_2,x2_1,x2_2])
#merged_vector = concatenate([x1_1,x1_2])

x = Flatten()(merged_vector)
x = BatchNormalization()(x)
x = Dense(64,activation='relu',kernel_initializer=random_normal(mean=0,stddev=1),bias_initializer=random_normal(mean=0,stddev=1))(x)
#x = Dropout(0.5)(x)
#x = Dense(128,activation='relu',kernel_initializer=random_normal(mean=0,stddev=1),bias_initializer=random_normal(mean=0,stddev=1))(x)
#x = Dropout(0.5)(x)
main_output = Dense(1,activation='sigmoid',kernel_initializer=random_normal(mean=0,stddev=1),bias_initializer=random_normal(mean=0,stddev=1))(x)

#from keras import backend as K
#def mse_l2(y_true,y_pred):
#	return K.mean(K.square(y_pred - y_true + 0.0001 * (K.square(y_pred - y_true))),axis=-1)

model = Model(inputs=[mat_1,mat_2],outputs=main_output)
#model = Model(inputs=mat_1,outputs=main_output)
model.compile(loss='mse',optimizer='sgd')
early_stopping = EarlyStopping(monitor='val_loss',patience=100)

start = time.time()

history = model.fit([x_train_1,x_train_2],y_train,batch_size=32,epochs=1000,validation_data=[[x_test_1,x_test_2],y_test],callbacks=[early_stopping])

end = time.time()
dur = end - start
print dur
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
plt.savefig("%s_loss.png" % name)

model.save('%s_model.h5' % name)
