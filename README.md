# SeqConv


##### The goal of this model is to predict the binding probability of a specific transcription factor based only on DNA sequence information of the binding region by means of convolutional neural network. Below is an overview of the strcture. 

<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/Figure1.png" width="1000" height="400" align=center />

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
##### For usage, one only needs to provide a TF narrowPeak file and genome file in FASTA format. In this demo, a narrowPeak file from maize ChIP-seq data is used.

    python prepare.py zm_tf.narrowPeak zm.fa zm_tf_train.txt
    python generator.py zm_tf.narrowPeak
    python SeqConv.py zm_tf_train.txt 
    python motifGen.py zm_tf
    Rscript plot_motif.r zm_tf
    
##### After training, there will be images showing the loss error and val_loss error during the training step, as well as the ROC curve showing the precision of the trained model. Additional steps are required to extract and plot the binding motif from convolution layer. And the trained model can be easily transfered to predict TFBS in other plants with high accuracy. 
<img src="https://raw.githubusercontent.com/shenwei19/SeqConv/master/imgs/Figure2.png" width=1000 height=400 align=center />


