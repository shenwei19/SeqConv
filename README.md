![](https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/logo.png)
## SeqConv
***

##### The goal of this model is to predict the binding probability of a specific transcription factor based only on DNA sequence information of the binding region by means of convolutional neural network. Below is an overview of the strcture. 

<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/Figure1.png" width="600" height="400" align=center />

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
<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/Figure2.png" height=300 align=center />

##### And the motif of this transcription factor
<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/test_motif.png" height=300 width=400 align=center />


