
import numpy as np
import sys
try:
    glb_dct = sys.modules['__main__'].__dict__
    THRESHOLD = glb_dct["THRESHOLD"]
except:
    THRESHOLD = 0
def prep_clf(obs,pre, threshold=THRESHOLD):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN 
    '''
    #根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=THRESHOLD):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold=THRESHOLD):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=THRESHOLD):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)

def FSC(obs, pre, threshold=THRESHOLD):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))


def TS(obs, pre, threshold=THRESHOLD):
    
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return hits/(hits + falsealarms + misses) 

def ETS(obs, pre, threshold=THRESHOLD):
    '''
    ETS - Equitable Threat Score
    details in the paper:
    Winterrath, T., & Rosenow, W. (2007). A new module for the tracking of
    radar-derived precipitation with model-derived winds.
    Advances in Geosciences,10, 77–83. https://doi.org/10.5194/adgeo-10-77-2007
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: ETS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)
    num = (hits + falsealarms) * (hits + misses)
    den = hits + misses + falsealarms + correctnegatives
    Dr = num / den

    ETS = (hits - Dr) / (hits + misses + falsealarms - Dr)

    return ETS

def FAR(obs, pre, threshold=THRESHOLD):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms) 
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)

def MAR(obs, pre, threshold=THRESHOLD):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)


def POD(obs, pre, threshold=THRESHOLD):
    '''
    func : 计算命中率 hits / (hits + misses)
    pod - Probability of Detection
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: PDO value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return hits / (hits + misses)

def BIAS(obs, pre, threshold = THRESHOLD):
    '''
    func: 计算Bias评分: Bias =  (hits + falsealarms)/(hits + misses) 
    	  alias: (TP + FP)/(TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''    
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return (hits + falsealarms) / (hits + misses)

def HSS(obs, pre, threshold=THRESHOLD):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: HSS value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    HSS_num = 2 * (hits * correctnegatives - misses * falsealarms)
    HSS_den = (misses**2 + falsealarms**2 + 2*hits*correctnegatives +
               (misses + falsealarms)*(hits + correctnegatives))

    return HSS_num / HSS_den

def BSS(obs, pre, threshold=THRESHOLD):
    '''
    BSS - Brier skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: BSS value
    '''
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs - pre) ** 2))

# def MAE(obs, pre):
#     """
#     Mean absolute error
#     Args:
#         obs (numpy.ndarray): observations
#         pre (numpy.ndarray): prediction
#     Returns:
#         float: mean absolute error between observed and simulated values
#     """
#     obs = np.where(obs != np.nan).flatten()
#     pre = np.where(pre != np.nan).flatten()

#     return np.mean(np.abs(pre - obs))


def RMSE(obs, pre):
    """
    Root mean squared error
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
    Returns:
        float: root mean squared error between observed and simulated values
    """
    obs = obs.flatten()
    pre = pre.flatten()

    return np.sqrt(np.mean((obs- pre) ** 2))

# print(scene2.shape[0])
# a=scene2[0,...]


def calculate(set_obs,set_pre):
    #glb_dct = sys.modules['__main__'].__dict__
    #global THRESHOLD
    #THRESHOLD = glb_dct["THRESHOLD"]
    #print(THRESHOLD)
    rmse = 0
    ts = 0
    # ets = 0
    far = 0
    # mar = 0
    pod = 0
    # bias = 0
    hss = 0
    # bss = 0
    num = set_obs.shape[0]
    counter_nants = 0
    counter_nanrmse = 0
    counter_nanfar = 0
    # counter_nanmar = 0
    counter_nanpod = 0
    counter_nanhss = 0
    # counter_nanbss = 0
    for i in range(num):
        rmse1 = RMSE(set_obs[i,...],set_pre[i,...])
        ts1 = TS(set_obs[i,...],set_pre[i,...])
        # ets1 = ETS(set_obs[i,...],set_pre[i,...])
        far1 = FAR(set_obs[i,...],set_pre[i,...])
        # mar1 = MAR(set_obs[i,...],set_pre[i,...])
        pod1 = POD(set_obs[i,...],set_pre[i,...])
        # bias1 = BIAS(set_obs[i,...],set_pre[i,...])
        hss1 = HSS(set_obs[i,...],set_pre[i,...])
        # bss1 = BSS(set_obs[i,...],set_pre[i,...])
        rmse+=rmse1
        if(np.isnan(rmse1)):
            counter_nanrmse+=1
            # print(set_obs[0,...])
            # print(set_pre[0,...])
            
        else:
            rmse1+=rmse1
        if(np.isnan(ts1)):
            counter_nants+=1
        else:
            ts+=ts1
        # if(np.isnan(ets1)):
        #     counter_nanets+=1
        # else:
        #     ets+=ets1
        if(np.isnan(far1)):
            counter_nanfar+=1
        else:
            far+=far1
        # if(np.isnan(mar1)):
        #     counter_nanmar+=1
        # else:
        #     mar+=mar1
        if(np.isnan(pod1)):
            counter_nanpod+=1
        else:
            pod+=pod1
        if(np.isnan(hss1)):
            counter_nanhss+=1
        else:
            hss+=hss1
        # if(np.isnan(bss1)):
        #     counter_nanbss+=1
        # else:
        #     bss+=bss1
        # bias+=bias1
            
    # rmse=rmse/num
    # ts=ts/(num-counter_nants)
    if (num-counter_nants)!=0:
        ts=ts/(num-counter_nants)
    else:
        ts= 0    
    if (num-counter_nanrmse)!=0:
        rmse= rmse/(num-counter_nanrmse)
    else:
        rmse= 0    
    # ets=ets/(num-counter_nanets)
    if (num-counter_nanfar)!=0:
        far= far/(num-counter_nanfar)
    else:
        far= 0
    # if (num-counter_nanmar)!=0:    
    #     mar= mar/(num-counter_nanmar)
    # else:
    #     mar=0
    if (num-counter_nanpod)!=0:
        pod= pod/(num-counter_nanpod)
    else:
        pod=0
    # bias= bias/realnum
    if (num-counter_nanhss)!=0:
        hss= hss/(num-counter_nanhss)
    else:
        hss=0
    
    # bss= bss/(num-counter_nanbss)
    return rmse,ts,far,pod,hss
def calculatepod(set_obs,set_pre):
    counter_nanrmse = 0
    #glb_dct = sys.modules['__main__'].__dict__
    #global THRESHOLD
    #THRESHOLD = glb_dct["THRESHOLD"]
    #print(THRESHOLD)
    rmse = 0
    num = set_obs.shape[0]
    # counter_nanbss = 0
    for i in range(num):
        rmse1 = POD(set_obs[i,...],set_pre[i,...])
        rmse+=rmse1
        if(np.isnan(rmse1)):
            counter_nanrmse+=1
            
        else:
            rmse1+=rmse1
    if (num-counter_nanrmse)!=0:
        rmse= rmse/(num-counter_nanrmse)
    else:
        rmse= 0   
    return rmse

def calculatermse(set_obs,set_pre):
    counter_nanrmse = 0
    #glb_dct = sys.modules['__main__'].__dict__
    #global THRESHOLD
    #THRESHOLD = glb_dct["THRESHOLD"]
    #print(THRESHOLD)
    rmse = 0
    num = set_obs.shape[0]
    # counter_nanbss = 0
    for i in range(num):
        rmse1 = RMSE(set_obs[i,...],set_pre[i,...])
        rmse+=rmse1
        if(np.isnan(rmse1)):
            counter_nanrmse+=1
            
        else:
            rmse1+=rmse1
    if (num-counter_nanrmse)!=0:
        rmse= rmse/(num-counter_nanrmse)
    else:
        rmse= 0   
    return rmse