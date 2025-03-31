import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import z_AI_NTS as AN
import scipy
from scipy import signal



def Minmax(input_):
    scaler = MinMaxScaler()  # 0-1사이로 정규화  평균0.5 분산1
    res = scaler.fit_transform(np.array(input_).reshape(-1, 1))
    res = torch.Tensor(res).view(-1)
    return res


'''''
if __name__ == '__main__':
    minute=3
    start=501
    length=1001

    env=AN.ATS_Env("'NQ'")
    env.data_create(minute)


    VIX= Minmax(env.VIX)
    nas= Minmax(env.nas)

    input_=[VIX[start:length],nas[start:length]]

    dae=AN.DAE_(input_,2)
    dae.load()
    v_hat= dae.forward()
    
    # 데이터 처리
    data_list = []
    for dim_ in range(dae.dim):
        data__ = v_hat[0][dim_].view(-1).tolist()
        data_list.append(data__)

    # PPO2의 input 생성
    input_data = torch.Tensor(data_list)

    for dim in range(dae.dim):
        plt.plot(input_data[dim])
        print(input_data[dim],'DAE res data')
    plt.show()

''''' \

minute=3
start=500
length=1000

env=AN.ATS_Env("'NQ'")
env.data_create(minute)


VIX= Minmax(env.VIX)
nas= Minmax(env.nas)

input_=[VIX[start:length],nas[start:length]]

res= signal.resample_poly(VIX[start:length], 1, 0, axis=0, window=('kaiser', 5.0), padtype='constant', cval=None)
print(res)
plt.plot(res)
plt.show()

