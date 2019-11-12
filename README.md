![](https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/logo.png)
***

##### The goal of this model is to predict the binding probability of a specific transcription factor based only on DNA sequence information of the binding region by means of convolutional neural network. Below is an overview of the strcture. 

<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/CNN_model.png" width="600" height="400" align=center />

***
##### Prerequisites

###### python modules:  

    tensorflow or Theano   
    keras  
    sklearn  
    numpy  
    matplotlib  
    tqdm  
    
###### R package:  

    motifStack
    
***
##### For your usage, you only need to provide narrowPeak file, genome file, chromSize file. In this demo, a narrowPeak file from Arabidopsis DAP-seq data is adopted to show how to use this model

    python prepare.py TAIR10.fa test.narrowPeak test_train.txt
    python generator.py TAIR10.fa test.narrowPeak test_train.txt TAIR10.chromsize 20000
    python SeqConv.py test_train.txt test
    python motifGen.py test_train.txt test
    Rscript plot_motif.r test_seq.meme test
    
##### After training, you will see an image showing the loss error and val_loss error in the training step
<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/test_loss.png" height=300 align=center />

##### And the motif of this transcription factor
<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/test_motif.png" height=300 width=400 align=center />

***
##### If you want to build a model of your own,here are some tricks that might be helpful
>##### Use probability score as the last layer (output) rather than binding intensity, or the model will be hard to converge

>##### Too many hiden layers will cause overfitting and thus cannot report correct motif

>##### Dropout is not necessary in this architecture

>##### There are some other methods to extract motif. For example, substitute nucleotide sequentially and calculate the change of probability to decide the importance of nucleotide
