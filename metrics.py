import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")
from math import ceil
def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True

class Metrictor:
    def __init__(self, classNum):
        self.classNum = classNum
        self._reporter_ = {"Score":self.Score,
                           "MaF":self.MaF, "MiF":self.MiF, 
                           "ACC":self.ACC,
                           "MaAUC":self.MaAUC, "MiAUC":self.MiAUC, 
                           "MaMCC":self.MaMCC, "MiMCC":self.MiMCC,
                           "Precision":self.Precision,"Recall":self.Recall,
                           "SOV993":self.SOV993,"SOV998":self.SOV998}
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    def set_data(self, Y_prob_pre, Y):
        self.raw_Y_pre,self.raw_Y = Y_prob_pre.argmax(axis=-1),Y
        Y_prob_pre,Y = Y_prob_pre.reshape(-1,self.classNum),Y.reshape(-1)
        self.Y_prob_pre,self.Y = Y_prob_pre,Y
        self.Y_pre = self.Y_prob_pre.argmax(axis=1)
        self.N = len(self.Y)
    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        id2lab = np.array(id2lab)
        Yarr = np.zeros((self.N, self.classNum), dtype='int32')
        Yarr[list(range(self.N)),self.Y] = 1
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(self.classNum, self.Y_pre, self.Y, ignore=False)
        MCCi = (TPi*TNi - FPi*FNi) / (np.sqrt( (TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi) ) + 1e-10)
        Pi = TPi/(TPi+FPi+1e-10)
        Ri = TPi/(TPi+FNi+1e-10)
        Fi = 2*Pi*Ri/(Pi+Ri+1e-10)
        sortedIndex = np.argsort(id2lab)
        classRate = Yarr.sum(axis=0)[sortedIndex] / self.N
        id2lab,MCCi,Pi,Ri,Fi = id2lab[sortedIndex],MCCi[sortedIndex],Pi[sortedIndex],Ri[sortedIndex],Fi[sortedIndex]
        print("-"*28 + "MACRO INDICTOR" + "-"*28)
        print(f"{'':30}{'rate':<8}{'MCCi':<8}{'Pi':<8}{'Ri':<8}{'Fi':<8}")
        for i,c in enumerate(id2lab):
            print(f"{c:30}{classRate[i]:<8.2f}{MCCi[i]:<8.3f}{Pi[i]:<8.3f}{Ri[i]:<8.3f}{Fi[i]:<8.3f}")
        print("-"*70)
    def MaF(self):
        return F1(self.classNum,  self.Y_pre, self.Y, average='macro')
    def MiF(self):
        return F1(self.classNum,  self.Y_pre, self.Y, average='micro')
    def ACC(self):
        return ACC(self.classNum, self.Y_pre, self.Y)
    def MaMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='macro')
    def MiMCC(self):
        return MCC(self.classNum, self.Y_pre, self.Y, average='micro')
    def MaAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='macro')
    def MiAUC(self):
        return AUC(self.classNum, self.Y_prob_pre, self.Y, average='micro')
    def Precision(self):
        return Precision(self.classNum, self.Y_pre, self.Y, average='micro')
    def Recall(self):
        return Recall(self.classNum, self.Y_pre, self.Y, average='micro')
    def SOV993(self):
        return SOV993(self.Y_pre,self.Y)
    def SOV998(self):
        return SOV998(self.Y_pre,self.Y)
    def Score(self):
        res = 0
        for y_pre,y in zip(self.raw_Y_pre,self.raw_Y):
            res += skmetrics.f1_score(y, y_pre, average='macro')
        return res/len(self.raw_Y)


def _TPiFPiTNiFNi(classNum, Y_pre, Y, ignore=True):
    Yarr, Yarr_pre = np.zeros((len(Y), classNum), dtype='int32'), np.zeros((len(Y), classNum), dtype='int32')
    
    Yarr[list(range(len(Y))),Y] = 1
    Yarr_pre[list(range(len(Y))),Y_pre] = 1
    if ignore:
        isValid = (Yarr.sum(axis=0) + Yarr_pre.sum(axis=0))>0
        Yarr,Yarr_pre = Yarr[:,isValid],Yarr_pre[:,isValid]
    TPi = np.array([Yarr_pre[:,i][Yarr[:,i]==1].sum() for i in range(Yarr.shape[1])], dtype='float32')
    
    FPi = Yarr_pre.sum(axis=0) - TPi
    TNi = (1^Yarr).sum(axis=0) - FPi
    FNi = Yarr.sum(axis=0) - TPi
    return TPi,FPi,TNi,FNi

def ACC(classNum, Y_pre, Y):
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    print(TPi.sum() / len(Y))
    return TPi.sum() / len(Y)

def AUC(classNum, Y_prob_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    Yarr = np.zeros((len(Y), classNum), dtype='int32')
    Yarr[list(range(len(Y))),Y] = 1
    return skmetrics.roc_auc_score(Yarr, Y_prob_pre, average=average)

def MCC(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        TP,FP,TN,FN = TPi.sum(),FPi.sum(),TNi.sum(),FNi.sum()
        MiMCC = (TP*TN - FP*FN) / (np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) ) + 1e-10)
        return MiMCC
    else:
        MCCi = (TPi*TNi - FPi*FNi) / (np.sqrt((TPi+FPi)*(TPi+FNi)*(TNi+FPi)*(TNi+FNi)) + 1e-10)
        return MCCi.mean()

def Precision(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiP = TPi.sum() / (TPi.sum() + FPi.sum() + 1e-10)
        
        return MiP
    else:
        Pi = TPi/(TPi+FPi+1e-10)
        print(Pi.mean())
        return Pi.mean()

def Recall(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
    if average=='micro':
        MiR = TPi.sum() / (TPi.sum() + FNi.sum() + 1e-10)
        return MiR
        
    else:
        Ri = TPi/(TPi + FNi + 1e-10)
        print(Ri.mean())
        return Ri.mean()

def F1(classNum, Y_pre, Y, average='micro'):
    assert average in ['micro', 'macro']
    if average=='micro':
        MiP,MiR = Precision(classNum, Y_pre, Y, average='micro'),Recall(classNum, Y_pre, Y, average='micro')
        MiF = 2*MiP*MiR/(MiP+MiR+1e-10)
        return MiF
    else:
        TPi,FPi,TNi,FNi = _TPiFPiTNiFNi(classNum, Y_pre, Y)
        Pi,Ri = TPi/(TPi + FPi + 1e-10),TPi/(TPi + FNi + 1e-10)
        Fi = 2*Pi*Ri/(Pi+Ri+1e-10)
        return Fi.mean()

def SOV993(Y_pre, Y):
    """
    计算二级结构预测的Sov评估指标。

    Args:
        Y (str): 观测到的二级结构字符串。
        Y_pre (str): 预测的二级结构字符串。

    Returns:
        float: Sov评估指标得分。
    """

    def len_segment(segment):
        """计算片段的长度。"""
        return segment[1] - segment[0] + 1

    def find_overlapping_pairs(Y_segments, Y_pre_segments):
        """查找观测到的片段和预测的片段之间的重叠对。"""
        overlapping_pairs = []
        for s1 in Y_segments:
            for s2 in Y_pre_segments:
                if s1[1] >= s2[0] and s2[1] >= s1[0]:
                    overlapping_pairs.append((s1, s2))
        return overlapping_pairs

    def calculate_sov_for_state(state):
        """计算给定状态的Sov得分和归一化因子N。"""
        Y_segments = [seg for seg in Y if seg[2] == state]
        Y_pre_segments = [seg for seg in Y_pre if seg[2] == state]

        overlapping_pairs = find_overlapping_pairs(Y_segments, Y_pre_segments)
        non_overlapping_Y = [s for s in Y_segments if s not in [p[0] for p in overlapping_pairs]]

        sov_state = 0
        N_state = sum(len_segment(s) for s in [p[0] for p in overlapping_pairs] + non_overlapping_Y)

        for s1, s2 in overlapping_pairs:
            maxov = max(s1[1], s2[1]) - min(s1[0], s2[0]) + 1
            minov = min(s1[1], s2[1]) - max(s1[0], s2[0]) + 1
            delta = min(maxov - minov, minov, ceil(len_segment(s1) / 2), ceil(len_segment(s2) / 2))
            sov_state += (minov + delta) / maxov * len_segment(s1)

        if N_state > 0:
            sov_state = 100 * sov_state / N_state
        return sov_state, N_state

    def convert_to_segments(structure_str):
        """将二级结构字符串转换为 (start, end, state) 格式的片段列表。"""
        segments = []
        start = 0
        prev_state = structure_str[0]
        for i in range(1, len(structure_str)):
            if structure_str[i] != prev_state:
                segments.append((start, i - 1, prev_state))
                start = i
                prev_state = structure_str[i]
        segments.append((start, len(structure_str) - 1, prev_state))
        return segments
    def Map(Y):
      mapping = {0: 'E', 1: 'H', 2: 'C'}
      Y = ''.join([mapping.get(item.item(), '') for item in Y])
      return Y
    Y=Map(Y)
    
    Y_pre=Map(Y_pre)
    Y_str=Y
    Y = convert_to_segments(Y)
    Y_pre = convert_to_segments(Y_pre)
 
    # 计算每个状态的Sov得分和归一化因子N
    sov_H, N_H = calculate_sov_for_state('H') if 'H' in Y_str else (0, 0)
    sov_E, N_E = calculate_sov_for_state('E') if 'E' in Y_str else (0, 0)
    sov_C, N_C = calculate_sov_for_state('C') if 'C' in Y_str else (0, 0)


    # 计算总的归一化因子N和Sov评估指标
    N = N_H + N_E + N_C
    existing_states = sum(1 for state in ['H', 'E', 'C'] if state in Y_str)
    SOV993 = (sov_H + sov_E + sov_C) / existing_states if existing_states > 0 else 0

    return SOV993
def SOV998(Y_pre, Y):
    """
    计算二级结构预测的Sov评估指标。

    Args:
        Y (str): 观测到的二级结构字符串。
        Y_pre (str): 预测的二级结构字符串。

    Returns:
        float: Sov评估指标得分。
    """

    def len_segment(segment):
        """计算片段的长度。"""
        return segment[1] - segment[0] + 1

    def find_overlapping_pairs(Y_segments, Y_pre_segments):
        """查找观测到的片段和预测的片段之间的重叠对。"""
        overlapping_pairs = []
        for s1 in Y_segments:
            for s2 in Y_pre_segments:
                if s1[1] >= s2[0] and s2[1] >= s1[0]:
                    overlapping_pairs.append((s1, s2))
        return overlapping_pairs

    def calculate_sov_for_state(state):
        """计算给定状态的Sov得分和归一化因子N。"""
        Y_segments = [seg for seg in Y if seg[2] == state]
        Y_pre_segments = [seg for seg in Y_pre if seg[2] == state]

        overlapping_pairs = find_overlapping_pairs(Y_segments, Y_pre_segments)
        non_overlapping_Y = [s for s in Y_segments if s not in [p[0] for p in overlapping_pairs]]

        sov_state = 0
        N_state = sum(len_segment(s) for s in [p[0] for p in overlapping_pairs] + non_overlapping_Y)

        for s1, s2 in overlapping_pairs:
            maxov = max(s1[1], s2[1]) - min(s1[0], s2[0]) + 1
            minov = min(s1[1], s2[1]) - max(s1[0], s2[0]) + 1
            delta = min(maxov - minov, minov, ceil(len_segment(s1) / 2), ceil(len_segment(s2) / 2))
            sov_state += (minov + delta) / maxov * len_segment(s1)

        if N_state > 0:
            sov_state = 100 * sov_state / N_state
        return sov_state, N_state

    def convert_to_segments(structure_str):
        """将二级结构字符串转换为 (start, end, state) 格式的片段列表。"""
        segments = []
        start = 0
        prev_state = structure_str[0]
        for i in range(1, len(structure_str)):
            if structure_str[i] != prev_state:
                segments.append((start, i - 1, prev_state))
                start = i
                prev_state = structure_str[i]
        segments.append((start, len(structure_str) - 1, prev_state))
        return segments
    def Map(Y):
      mapping = {0: 'E', 1: 'H', 2: 'C',3:"S",4:"B",5:"T",6:"G",7:"I"}
      Y = ''.join([mapping.get(item.item(), '') for item in Y])
      return Y
    Y=Map(Y)
    Y_pre=Map(Y_pre)
    Y_str=Y
    Y = convert_to_segments(Y)
    Y_pre = convert_to_segments(Y_pre)
 
    # 计算每个状态的Sov得分和归一化因子N
    sov_H, N_H = calculate_sov_for_state('H') if 'H' in Y_str else (0, 0)
    sov_E, N_E = calculate_sov_for_state('E') if 'E' in Y_str else (0, 0)
    sov_C, N_C = calculate_sov_for_state('C') if 'C' in Y_str else (0, 0)
    sov_S, N_S = calculate_sov_for_state('S') if 'S' in Y_str else (0, 0)
    sov_B, N_B = calculate_sov_for_state('B') if 'B' in Y_str else (0, 0)
    sov_T, N_T = calculate_sov_for_state('T') if 'T' in Y_str else (0, 0)
    sov_G, N_G = calculate_sov_for_state('G') if 'G' in Y_str else (0, 0)
    sov_I, N_I = calculate_sov_for_state('I') if 'I' in Y_str else (0, 0)


    # 计算总的归一化因子N和Sov评估指标
    N = N_H + N_E + N_C
    existing_states = sum(1 for state in ['H', 'E', 'C',"S","B","T","G","I"] if state in Y_str)
    SOV998 = (sov_H + sov_E + sov_C+sov_S + sov_B + sov_T+sov_G + sov_I) / existing_states if existing_states > 0 else 0

    return SOV998
    
