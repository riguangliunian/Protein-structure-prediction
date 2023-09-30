# Protein-structure-prediction
## Introduction
This code is used to implement the distilled improved TCN-BiRNN-MLP and perform protein secondary structure prediction on it.
we propose a new TCN layer and a new combined model, improved TCN-BIRNN-MLP, word vector extraction of one-hot coding features and physicochemical properties of proteins by word2vec disambiguation, using knowledge distillation to allow student models to learn the rich features of the ProtT5-XL-UniRef teacher model and multiple datasets for protein octapeptide and tripeptide prediction. The current dataset was derived from the classical datasets TS115 and CB513 and protein primary and secondary structure data selected from the PDB Protein Data Bank

## Datasets
First, two classical datasets, TS115 and CB513 , whose small amounts of data can effectively respond to the model's effect, are used; at the same time, 15,078 protein data points from 2018-06-6 to 2020 are introduced from the PDB database

## Requirements
gensim==4.3.2  
numpy==1.23.5  
pandas==1.5.3  
scikit-learn==1.2.2  
scipy==1.11.2  
tensorflow==2.13.0  
keras==2.13.1  
torch==2.0.1+cu118  
python==3.10

we need to configure a GPU with about 40G memory to fully realize the running of all codes.
