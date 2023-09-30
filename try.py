# 导入相关类
from utils import *
from student import *
from teacher import *
from distillation import *

# 初始化模型对象
model = improvedModel(classNum=dataClass.classNum, embedding=dataClass.vector['embedding'], feaEmbedding=dataClass.vector['feaEmbedding'],useFocalLoss=True, device=torch.device('cuda'))
# 开始训练
model.cv_train( dataClass, trainSize=64, batchSize=64, epoch=1000, stopRounds=100, earlyStop=30, saveRounds=1,
                savePath='model/FinalModel', lr=3e-4, augmentation=0.1, kFold=3)
