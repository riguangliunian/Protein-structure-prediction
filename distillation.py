from teacher import *
from student import *
import random
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn.functional as F

seed = 42
random.seed(seed)

#开始进行知识蒸馏算法
#model1小模型；
#蒸馏温度
T=7
hard_loss=nn.CrossEntropyLoss()
alpha=0.3
soft_loss=nn.KLDivLoss(reduction="batchmean")
optim=torch.optim.Adam(model.parameters(),lr=0.0001)

all_values = [value for value in sorted_dict.values()]
Y_hat = [[value] for value in all_values]

for i in range(len(Y_hat)):
# 获取NumPy数组
  arr = Y_hat[i]
# 使用 ndim 属性查看数组的维度
  dimensions = np.ndim(arr)
  if dimensions==1:
    print(i)
    
formatted_data = [[list(arr) for arr in sublist] for sublist in Y_hat]

new = []

for i in range(len(Y_true)):
    flat_list = [item for sublist in Y_true[i] for item in sublist]
    new.append(flat_list)
new2 = []

for i in range(len(Y_hat)):
    flat_list = [item for sublist in Y_hat[i] for item in sublist]
    new2.append(flat_list)
    
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ParallelSequenceDataLoader:
    def __init__(self, data1, data2, batch_size):
        assert len(data1) == len(data2), "Data lengths must be the same"

        self.data1 = data1
        self.data2 = data2
        self.batch_size = batch_size
        self.num_samples = len(data1)
        self.indices = list(range(self.num_samples))
        self.current_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch_data1 = [self.data1[idx] for idx in batch_indices]
        batch_data2 = [self.data2[idx] for idx in batch_indices]

        self.current_idx += self.batch_size

        # Padding
        max_length1 = max([len(seq) for seq in batch_data1])
        max_length2 = max([len(seq) for seq in batch_data2])

        padded_batch_data1 = [seq + [0] * (max_length1 - len(seq)) for seq in batch_data1]
        padded_batch_data2 = [seq + [0] * (max_length2 - len(seq)) for seq in batch_data2]

        # Convert to tensors
        padded_batch_tensor1 = torch.tensor(padded_batch_data1)
        padded_batch_tensor2 = torch.tensor(padded_batch_data2)

        return padded_batch_tensor1, padded_batch_tensor2

batch_size = 64

data_loader = ParallelSequenceDataLoader(new, new2, batch_size)

# Get a batch of X and X2 as Tensors
batch_X, batch_X2 = next(data_loader)

##第二种蒸馏
seq_len = []

# 遍历 Y_hat 列表中的每个 NumPy 数组
for i in range(5000):  # 请注意这里应该是 range(5000)，不是 range(0:4999)
    # 计算当前 NumPy 数组的长度
    array_lengths = [len(arr) for arr in Y_true[i]]
    seq_len.extend(array_lengths)

# 输出结果
print("Lengths of arrays:", seq_len)

# 找到最大的子数组长度
max_length = max(seq_len)

# 创建一个列表来存储转换后的 PyTorch 张量
tensor_list = []

# 将每个 NumPy 数组转换为 PyTorch 张量并保存到列表中
for arr in Y_hat:
    tensor_arr = torch.tensor(arr, dtype=torch.int64)
    tensor_list.append(tensor_arr)

# 参数设置
batch_size = 64
class_num = 9   # 替换为你的实际类别数量

def process_tensor_list(tensor_list, class_num):
    processed_list = []

    for prediction in tensor_list:
        seq_len = len(prediction)  # 获取当前张量的长度

        # 将预测结果 reshape 成 (1, seq_len)
        prediction_reshaped = prediction.view(1, -1)  # 使用 -1 自动计算维度

        # 将预测结果转换为 one-hot 编码
        prediction_onehot = F.one_hot(prediction_reshaped, num_classes=class_num)

        processed_list.append(prediction_onehot)

    return processed_list

processed_tensor_list = process_tensor_list(tensor_list, class_num)

def process_and_softmax(tensor_list, class_num):
    processed_list = []

    for prediction in tensor_list:
        seq_len = len(prediction)  # 获取当前张量的长度

        # 将预测结果 reshape 成 (1, seq_len)
        prediction_reshaped = prediction.view(1, -1)  # 使用 -1 自动计算维度

        # 将预测结果转换为 one-hot 编码
        prediction_onehot = F.one_hot(prediction_reshaped, num_classes=class_num)

        # 将 one-hot 编码的张量转换为浮点数
        prediction_onehot_float = prediction_onehot.float()

        # 使用 softmax 函数将预测值转换为概率分布
        probabilities = F.softmax(prediction_onehot_float, dim=2)

        processed_list.append(probabilities)

    return processed_list

# 示例使用
class_num = 9

# 调用函数进行处理和 softmax 转换
processed_softmax_list = process_and_softmax(tensor_list, class_num)

# 输出结果
for probabilities in processed_softmax_list:
    print(probabilities)

# 找到最大的序列长度
max_seq_len = max([tensor.size(2) for tensor in processed_softmax_list])

# 手动进行 padding
padded_tensors = []
for tensor in processed_softmax_list:
    pad_size = max_seq_len - tensor.size(2)
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size))
    padded_tensors.append(padded_tensor)

# 创建一个 TensorDataset，用于构建 DataLoader
dataset = TensorDataset(*padded_tensors)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

############蒸馏实现
class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
                 savePath='model'):
        kf = KFold(n_splits=kFold)
        validRes = []
        for i,(trainIndices,validIndices) in enumerate(kf.split(range(dataClass.totalSampleNum))):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,self.validSampleNum = len(trainIndices),len(validIndices)
            dataClass.describe()
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,augmentation,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
              savePath='model'):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device, augmentation=augmentation)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device, augmentation=augmentation)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                # 蒸馏
                student_loss = self.calculate_loss(X,Y)
                # 计算教师模型的loss
                yh,yt=next(data_loader)
                teacher_loss=hard_loss(yh.float(),yt.float())
                # 综合教师和学生损失来进行反向传播和优化
                optimizer.zero_grad()
                total_loss = alpha * teacher_loss + (1 - alpha) * student_loss
                total_loss.backward()
                optimizer.step()
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                #Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                #metrictor.set_data(Y_pre, Y)
                #print(f'[Total Train]',end='')
                #metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2secItem)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
            stateDict['seqItem2id'],stateDict['id2seqItem'] = dataClass.seqItem2id,dataClass.id2seqItem
            stateDict['secItem2id'],stateDict['id2secItem'] = dataClass.secItem2id,dataClass.id2secItem
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList,dataClass.validIdList = parameters['trainIdList'],parameters['validIdList']
            dataClass.seqItem2id,dataClass.id2seqItem = parameters['seqItem2id'],parameters['id2seqItem']
            dataClass.secItem2id,dataClass.id2secItem = parameters['secItem2id'],parameters['id2secItem']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
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
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate
    
##第二种蒸馏
class BaseClassifier:
    def __init__(self):
        pass
    def calculate_y_logit(self, X, XLen):
        pass
    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
                 optimType='Adam', lr=0.001, weightDecay=0, kFold=5, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
                 savePath='model'):
        kf = KFold(n_splits=kFold)
        validRes = []
        for i,(trainIndices,validIndices) in enumerate(kf.split(range(dataClass.totalSampleNum))):
            print(f'CV_{i+1}:')
            self.reset_parameters()
            dataClass.trainIdList,dataClass.validIdList = trainIndices,validIndices
            dataClass.trainSampleNum,self.validSampleNum = len(trainIndices),len(validIndices)
            dataClass.describe()
            res = self.train(dataClass,trainSize,batchSize,epoch,stopRounds,earlyStop,saveRounds,augmentation,optimType,lr,weightDecay,
                             isHigherBetter,metrics,report,f"{savePath}_cv{i+1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)
    def train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1, augmentation=0.05,
              optimType='Adam', lr=0.001, weightDecay=0, isHigherBetter=True, metrics="Score", report=["ACC", "MaF", "Score"],
              savePath='model'):
        assert batchSize%trainSize==0
        metrictor = Metrictor(dataClass.classNum)
        self.stepCounter = 0
        self.stepUpdate = batchSize//trainSize
        optimizer = torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', device=self.device, augmentation=augmentation)
        itersPerEpoch = (dataClass.trainSampleNum+trainSize-1)//trainSize
        mtc,bestMtc,stopSteps = 0.0,0.0,0
        if dataClass.validSampleNum>0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='valid', device=self.device, augmentation=augmentation)
        st = time.time()
        for e in range(epoch):
            for i in range(itersPerEpoch):
                self.to_train_mode()
                X,Y = next(trainStream)
                # 蒸馏
                student_loss = self.calculate_loss(X,Y)
                # 计算教师模型的loss
                student_preds = self.calculate_y_logit(X)
                teacher_preds=next(data_iter)
                distillation_loss = soft_loss(
                F.softmax(student_preds/temp, dim=1),
                F.softmax(teacher_preds/temp, dim=1))
                # 综合教师和学生损失来进行反向传播和优化
                optimizer.zero_grad()
                total_loss = alpha * teacher_loss + (1 - alpha) * student_loss
                total_loss.backward()
                optimizer.step()
                if stopRounds>0 and (e*itersPerEpoch+i+1)%stopRounds==0:
                    self.to_eval_mode()
                    print(f"After iters {e*itersPerEpoch+i+1}: [train] loss= {loss:.3f};", end='')
                    if dataClass.validSampleNum>0:
                        X,Y = next(validStream)
                        loss = self.calculate_loss(X,Y)
                        print(f' [valid] loss= {loss:.3f};', end='')
                    restNum = ((itersPerEpoch-i-1)+(epoch-e-1)*itersPerEpoch)*trainSize
                    speed = (e*itersPerEpoch+i+1)*trainSize/(time.time()-st)
                    print(" speed: %.3lf items/s; remaining time: %.3lfs;"%(speed, restNum/speed))
            if dataClass.validSampleNum>0 and (e+1)%saveRounds==0:
                self.to_eval_mode()
                print(f'========== Epoch:{e+1:5d} ==========')
                #Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
                #metrictor.set_data(Y_pre, Y)
                #print(f'[Total Train]',end='')
                #metrictor(report)
                print(f'[Total Valid]',end='')
                Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                print('=================================')
                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                    print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save("%s.pkl"%savePath, e+1, bestMtc, dataClass)
                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps>=earlyStop:
                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                        break
        self.load("%s.pkl"%savePath)
        os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
        print(f'============ Result ============')
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='train', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Train]',end='')
        metrictor(report)
        Y_pre,Y = self.calculate_y_prob_by_iterator(dataClass.one_epoch_batch_data_stream(trainSize, type='valid', device=self.device))
        metrictor.set_data(Y_pre, Y)
        print(f'[Total Valid]',end='')
        res = metrictor(report)
        metrictor.each_class_indictor_show(dataClass.id2secItem)
        print(f'================================')
        return res
    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()
    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            stateDict['trainIdList'],stateDict['validIdList'] = dataClass.trainIdList,dataClass.validIdList
            stateDict['seqItem2id'],stateDict['id2seqItem'] = dataClass.seqItem2id,dataClass.id2seqItem
            stateDict['secItem2id'],stateDict['id2secItem'] = dataClass.secItem2id,dataClass.id2secItem
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            dataClass.trainIdList,dataClass.validIdList = parameters['trainIdList'],parameters['validIdList']
            dataClass.seqItem2id,dataClass.id2seqItem = parameters['seqItem2id'],parameters['id2seqItem']
            dataClass.secItem2id,dataClass.id2secItem = parameters['secItem2id'],parameters['id2secItem']
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))
    def calculate_y_prob(self, X):
        Y_pre = self.calculate_y_logit(X)
        return torch.softmax(Y_pre, dim=1)
    def calculate_y(self, X):
        Y_pre = self.calculate_y_prob(X)
        return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y):
        Y_logit = self.calculate_y_logit(X)
        return self.criterion(Y_logit, Y)
    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre,Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)
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
        YArr,Y_preArr = np.hstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('float32')
        return Y_preArr, YArr
    def calculate_y_by_iterator(self, dataStream):
        Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
        return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()
    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()
    def _train_step(self, X, Y, optimizer):
        self.stepCounter += 1
        if self.stepCounter<self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y)/self.stepUpdate
        loss.backward()
        if p:
            optimizer.step()
            optimizer.zero_grad()
        return loss*self.stepUpdate