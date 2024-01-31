import numpy as np
import torch,time,os,pickle
from torch import nn as nn
from Layer import *
from metrics import *
from typing import Iterable
from collections import Counter
from sklearn.model_selection import KFold

class improvedModel(BaseClassifier):
    def __init__(self, classNum, embedding, feaEmbedding, feaSize=64,
                 filterNum=128, contextSizeList=[1,9,81],
                 hiddenSize=512, num_layers=3,
                 hiddenList=[2048],
                 embDropout=0.2, BiGRUDropout=0.2, fcDropout=0.4,
                 useFocalLoss=False, weight=-1, device=torch.device("cuda:0")):
        self.textEmbedding = TextEmbedding( torch.tensor(embedding, dtype=torch.float),dropout=embDropout ).to(device)
        self.feaEmbedding = TextEmbedding( torch.tensor(feaEmbedding, dtype=torch.float),dropout=embDropout//2,name='feaEmbedding',freeze=True ).to(device)
        self.textCNN = TextTCN( feaSize, contextSizeList, filterNum ).to(device)
        self.textBiGRU = TextBiGRULSTM(len(contextSizeList)*filterNum, hiddenSize, num_layers=num_layers, dropout=BiGRUDropout).to(device)
        self.fcLinear = MLP(len(contextSizeList)*filterNum+hiddenSize*2, classNum, hiddenList, fcDropout).to(device)
        self.moduleList = nn.ModuleList([self.textEmbedding,self.feaEmbedding,self.textCNN,self.textBiGRU,self.fcLinear])
        self.classNum = classNum
        self.device = device
        self.feaSize = feaSize
        self.criterion = nn.CrossEntropyLoss() if not useFocalLoss else FocalCrossEntropyLoss(weight=weight).to(device)
    def calculate_y_logit(self, X):
        X = X['seqArr']
        X = torch.cat([self.textEmbedding(X),self.feaEmbedding(X)], dim=2) # => batchSize × seqLen × feaSize
        X_conved = self.textCNN(X) # => batchSize × seqLen × scaleNum*filterNum
        X_BiGRUed = self.textBiGRU(X_conved, None) # => batchSize × seqLen × hiddenSize*2
        X = torch.cat([X_conved,X_BiGRUed], dim=2) # => batchSize × seqLen × (scaleNum*filterNum+hiddenSize*2)
        return self.fcLinear(X) # => batchSize × seqLen × classNum
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=2)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=2)
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=2), YArr
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        Y = Y.reshape(-1)
        Y_logit = Y_logit.reshape(len(Y),-1)
        return self.criterion(Y_logit, Y)
    def calculate_y_prob_by_iterator(self, dataStream):
        YArr,Y_preArr = [],[]
        while True:
            try:
                X,Y = next(dataStream)
            except:
                break
            Y_pre,Y = self.calculate_y_prob(X).cpu().data.numpy(),Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr





# 初始化数据类
dataClass = DataClass('data1.txt', 'data2.txt', k=1, validSize=0.2, minCount=0)

trainStream = dataClass.random_batch_data_stream(batchSize=128, type='train', device="cuda", augmentation=0.5)

# 词向量预训练
dataClass.vectorize(method='char2vec', feaSize=23, sg=1)

# onehot+理化特征获取
dataClass.vectorize(method='feaEmbedding')

