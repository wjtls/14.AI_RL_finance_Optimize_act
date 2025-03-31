aimport torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ATS_Env:
    def __init__(self, data_name):
        self.data_name = data_name

        # 데이터 호출
        self.data_date = 0
        self.USD_date = 0
        self.VIX_date = 0
        self.Note_date = 0
        self.SNP_date = 0
        self.Gold_date = 0
        self.DAX_date = 0
        self.STOXX_date = 0
        self.Oil_date = 0
        self.AUD_date = 0
        self.CAD_date = 0
        self.Peso_date = 0
        self.HG_date = 0
        self.Dow_date = 0
        self.nas_date = 0

        self.data = 0
        self.USD = 0
        self.VIX = 0
        self.Note = 0
        self.SNP = 0
        self.Gold = 0
        self.DAX = 0
        self.STOXX = 0
        self.Oil = 0
        self.AUD = 0
        self.CAD = 0
        self.Peso = 0
        self.HG = 0
        self.Dow = 0
        self.nas = 0

    def time_Frame(self, data, minute):  # 분봉 출력
        try:
            price_data = [float(data_[0]) for data_ in data]

        except:
            price_data = [data_[0] for data_ in data]

        price_data = pd.Series(price_data)
        price_data.dropna(inplace=True)
        index_data = [t * minute for t in range(len(price_data)) if t * minute < len(price_data)]
        res = price_data[index_data].reset_index()[0]
        return res

    def data_pre(self, data_name,minute):
        #DB 데이터 호출
        connection = psycopg2.connect(dbname='postgres', user='postgres', password='snowai**', host='172.30.1.96', port='5432', sslmode='require')

        db = connection.cursor()
        db.execute("SELECT close FROM snowball.price_minute WHERE symbol="+data_name+" ORDER BY datetime ASC, Symbol ASC;")
        dt = db.fetchall()
        dt = self.time_Frame(dt, minute)

        db.execute("SELECT datetime FROM snowball.price_minute WHERE symbol="+data_name+" ORDER BY datetime ASC, Symbol ASC;")
        date = db.fetchall()
        date=self.time_Frame(date,minute)

        return dt,date

    def data_create(self,minute):
        self.data, self.data_date = self.data_pre(self.data_name,minute)
        self.USD, self.USD_date = self.data_pre("'DX'",minute)
        self.VIX, self.VIX_date = self.data_pre("'VX'",minute)
        self.Note, self.Note_date = self.data_pre("'ZN'",minute)
        self.SNP, self.SNP_date = self.data_pre("'ES'",minute)
        self.Gold, self.Gold_date = self.data_pre("'GC'",minute)
        self.DAX, self.DAX_date = self.data_pre("'FDX'",minute)
        self.STOXX, self.STOXX_date = self.data_pre("'FESX'",minute)
        self.Oil, self.Oil_date = self.data_pre("'CL'",minute)
        self.AUD, self.AUD_date = self.data_pre("'6A'",minute)
        self.CAD, self.CAD_date = self.data_pre("'6C'",minute)
        self.Peso, self.Peso_date = self.data_pre("'6M'",minute)
        self.HG, self.HG_date = self.data_pre("'HG'",minute)
        self.Dow, self.Dow_date = self.data_pre("'YM'",minute)
        self.nas, self.nas_date = self.data_pre("'NQ'",minute)

        #데이터 크기 맞춘다
        start_period=-min(len(self.data),len(self.USD),len(self.VIX),len(self.Note),len(self.SNP),len(self.Gold)
                          ,len(self.DAX),len(self.STOXX),len(self.Oil),len(self.AUD),len(self.CAD),len(self.Peso),len(self.HG)
                          ,len(self.Dow),len(self.nas))

        self.data=torch.Tensor(self.data[start_period:].tolist())
        self.USD = torch.Tensor(self.USD[start_period:].tolist())
        self.VIX = torch.Tensor(self.VIX[start_period:].tolist())
        self.Note = torch.Tensor(self.Note[start_period:].tolist())
        self.SNP = torch.Tensor(self.SNP[start_period:].tolist())
        self.DAX=torch.Tensor(self.DAX[start_period:].tolist())
        self.Gold = torch.Tensor(self.Gold[start_period:].tolist())
        self.STOXX = torch.Tensor(self.STOXX[start_period:].tolist())
        self.Oil = torch.Tensor(self.Oil[start_period:].tolist())
        self.AUD = torch.Tensor(self.AUD[start_period:].tolist())
        self.CAD = torch.Tensor(self.CAD[start_period:].tolist())
        self.Peso = torch.Tensor(self.Peso[start_period:].tolist())
        self.HG = torch.Tensor(self.HG[start_period:].tolist())
        self.Dow = torch.Tensor(self.Dow[start_period:].tolist())
        self.nas = torch.Tensor(self.nas[start_period:].tolist())

        self.data_date=self.data_date[start_period:]
        self.USD_date=self.USD_date[start_period:]
        self.VIX_date=self.VIX_date[start_period:]
        self.Note_date=self.Note_date[start_period:]
        self.SNP_date=self.SNP_date[start_period:]
        self.DAX_date=self.DAX_date[start_period:]
        self.Gold_date=self.Gold_date[start_period:]
        self.STOXX_date=self.STOXX_date[start_period:]
        self.Oil_date=self.Oil_date[start_period:]
        self.AUD_date=self.AUD_date[start_period:]
        self.CAD_date=self.CAD_date[start_period:]
        self.Peso_date=self.Peso_date[start_period:]
        self.HG_date=self.HG_date[start_period:]
        self.Dow_date=self.Dow_date[start_period:]
        self.nas_date=self.nas_date[start_period:]



class CDAE_(nn.Module):

    # Linear network를 사용할수도 있지만 Conv net 을 사용함으로
    # 여러 feature의 특징추출을 목적으로 생각했기에 CNN을 사용했다
    def __init__(self, input_, dim):
        nn.Module.__init__(self)


        #데이터 and 파라미터
        self.ori_input = input_
        self.input_ = self.CNN_observation(torch.cat(input_), dim)
        self.dim = dim
        self.out_dim = dim
        self.kernel_size = (1, 10)
        stride_ = 1

        self.encoder = nn.Sequential(nn.Conv2d(self.dim, 128, kernel_size=self.kernel_size, stride=stride_,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(128, 64, kernel_size=self.kernel_size, stride=stride_,padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool2d(3, stride=stride_),
                                     nn.ReLU(),
                                     nn.Conv2d(64, 32, kernel_size=self.kernel_size, stride=stride_),
                                     )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=self.kernel_size, stride=stride_),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=stride_),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 128, kernel_size=self.kernel_size, stride=stride_),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.out_dim, kernel_size=self.kernel_size, stride=stride_))



        self.check_point = os.path.join('DAE_weight')  # 저장위한 체크포인트
        self.optimizer = optim.Adam(self.parameters(), lr=3e-4, eps=1e-10)

    def load(self):
        self.load_state_dict(torch.load(self.check_point))

    def save(self):
        torch.save(self.state_dict(), self.check_point)

    def forward(self):
        res = self.encoder(self.input_)
        res = F.relu(res)
        res = self.decoder(res)
        res =res
        return res

    def train(self, epoch):  # DAE 실행후

        for t in range(epoch):
            v_hat = self.forward()

            with torch.no_grad():
                target = self.input_

            self.optimizer.zero_grad()
            loss = F.mse_loss(v_hat, target.detach())

            loss.backward()
            self.optimizer.step()

            self.save()  # 가중치 저장


            if t % 10 == 0:
                print(t + 1, '/', epoch, '스탭', loss, ': loss값')

            if loss < 5e-5:  # 로스가 일정수치 아래로 감소하면 학습멈춘다
                ('목표 loss 도달. overfitting 방지를위해 학습 정지')
                break

        print(v_hat,'v_hat')
        print(v_hat.size(),'vvv_hat')

        # 데이터 처리
        data_list = []
        for dim_ in range(self.dim):
            data__ = v_hat[0][dim_].view(-1).tolist()
            data_list.append(data__)

        # PPO2의 input 생성
        input_data = torch.Tensor(data_list)
        print('Denoise 완료.')

        #plot
        fig, ax = plt.subplots(3, 1, figsize=(10, 9))
        for dim_ in range(self.dim):
            ax[0].set_ylabel('input')
            ax[0].plot(self.ori_input[dim_][200:400])

            ax[1].set_ylabel('result DAE')
            ax[0].plot(input_data[dim_][200:400])

            ax[2].set_ylabel('test')
            ax[2].plot(input_data[dim_])
        plt.show()




        return input_data

    def LSTM_observation(self, input_, window, input_dim):  # 윈도우사이즈만큼 텐서형태로 출력 LSTM인풋 데이터로 만듦
        window_size = window
        data = []
        input_ = input_
        for k in range(len(input_[0]) - (window_size - 1)):
            for dim_idx in range(input_dim):
                data.append(input_[dim_idx][k:k + window_size])
        data = torch.Tensor(torch.cat(data).view(-1, input_dim, window))
        # batch first 일경우      (배치길이(총길이), 디멘션, 시퀀스길이(윈도우사이즈))

        return data

    def CNN_observation(self, input_, dim):  # CNN인풋 데이터로 만든다
        # 배치 크기 × dim × 높이(height) × 너비(widht,window size)의 크기의 텐서를 선언
        data = input_.view(1, dim, 1, -1)

        return data





def Minmax(input_):
    scaler = MinMaxScaler()  # 0-1사이로 정규화  평균0.5 분산1
    res = scaler.fit_transform(np.array(input_).reshape(-1, 1))
    res = torch.Tensor(res).view(-1)
    return res


if __name__ == '__main__':
    minute=3

    env=ATS_Env("'NQ'")
    env.data_create(minute)


    VIX= Minmax(env.VIX)
    nas= Minmax(env.nas)

    input_=[nas[:500]]

    dae=CDAE_(input_,1)
    dae.train(10000)




#https://stackoverflow.com/questions/63500337/denoising-linear-autoencoder-learns-to-output-a-constant-instead-of-denoising

class LinAutoencoder(nn.Module):
    def __init__(self, in_channels, K, B, z_dim, out_channels):
        super(LinAutoencoder, self).__init__()

        self.in_channels = in_channels
        self.K = K # number of samples per 2pi interval
        self.B = B # how many intervals
        self.out_channels = out_channels

        encoder_layers = []
        decoder_layers = []

        encoder_layers += [
            nn.Linear(in_channels * K * B, 2*z_dim, bias=True),
            nn.ReLU(),
            nn.Linear(2*z_dim, z_dim, bias=True),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim, bias=True),
            nn.ReLU()
        ]

        decoder_layers += [
            nn.Linear(z_dim, z_dim, bias=True),
            nn.ReLU(),
            nn.Linear(z_dim, 2*z_dim, bias=True),
            nn.ReLU(),
            nn.Linear(2*z_dim, out_channels * K * B, bias=True),
            nn.Tanh()
        ]


        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        batch_size = x.shape[0]
        x_flat = torch.flatten(x, start_dim=1)
        enc = self.encoder(x_flat)
        dec = self.decoder(enc)
        res = dec.view((batch_size, self.out_channels, self.K * self.B))
        return res