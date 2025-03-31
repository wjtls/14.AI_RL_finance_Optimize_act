#NTS 1m

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#정규화 및 전처리 계산
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import multiprocessing as mp
from multiprocessing import Process,Queue


import pandas as pd
import sqlite3
import pymysql
import psycopg2
import pprint


#시각화및 저장,계산
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import quantstats as qs

#API및 데이터 불러오기
import pyupbit as py
import csv

#기타
import datetime
import time
import copy
from shholiday import holiday2020 as hd
from datetime import  datetime

#크롤링
import requests
import xmltodict
import json
from pandas import json_normalize

import requests
from bs4 import BeautifulSoup
import FinanceDataReader as fd

# 시각화및 저장,계산
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import quantstats as qs
import time

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

# API및 데이터 불러오기
import pyupbit as py
import csv
from datetime import datetime
import requests


class NTS_Env:
    def __init__(self, data_name,data_count):
        self.data_name = data_name
        self.data_count=data_count

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
        db.execute("SELECT open FROM snowball.price_minute WHERE symbol="+data_name+" ORDER BY datetime ASC, Symbol ASC;")
        dt = db.fetchall()
        dt = self.time_Frame(dt, minute)

        db.execute("SELECT datetime FROM snowball.price_minute WHERE symbol="+data_name+" ORDER BY datetime ASC, Symbol ASC;")
        date = db.fetchall()
        date=self.time_Frame(date,minute)

        dt=dt[-self.data_count:]
        date=dt[-self.data_count:]
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

        self.data=self.data[start_period:]
        self.USD = self.USD[start_period:]
        self.VIX = self.VIX[start_period:]
        self.Note = self.Note[start_period:]
        self.SNP = self.SNP[start_period:]
        self.DAX=self.DAX[start_period:]
        self.Gold = self.Gold[start_period:]
        self.STOXX = self.STOXX[start_period:]
        self.Oil = self.Oil[start_period:]
        self.AUD = self.AUD[start_period:]
        self.CAD = self.CAD[start_period:]
        self.Peso = self.Peso[start_period:]
        self.HG = self.HG[start_period:]
        self.Dow = self.Dow[start_period:]
        self.nas = self.nas[start_period:]

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

    def reset(self):
        pass

    def MA(self, data, period):  # 이동 평균선 함수
        ma = data.rolling(period).mean()
        return ma

    def CCI(self, data, period):  # CCI 함수
        Aver = self.MA(data, period)

        nan_idx = period - 1

        # nan 제거 (인덱스 일치시킴)
        Aver = Aver.iloc[nan_idx:]
        data = data.iloc[nan_idx:]

        self.reset()  # 파라미터 초기화

        # CCI 계산
        p = np.abs(data - Aver)
        MD = p.rolling(period).mean()

        MD = MD[period - 1:]
        data = data[period - 1:]
        Aver = Aver[period - 1:]

        CCI_data = ((data - Aver) / (0.015 * (MD + (1e-1))))

        CCI_data = pd.Series(CCI_data.reset_index().drop('index', axis=1).values.reshape(-1))

        return CCI_data, nan_idx

    def TRI(self, vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8):

        # CCI 값 호출
        USD_CCI, nan1 = self.CCI(self.USD, CCI1)
        VIX_CCI, nan1 = self.CCI(self.VIX, CCI1)
        NOTE_CCI, nan1 = self.CCI(self.Note, CCI1)
        Gold_CCI, nan1 = self.CCI(self.Gold, CCI1)

        USD_CCI2, nan2 = self.CCI(self.USD, CCI2)
        VIX_CCI2, nan2 = self.CCI(self.VIX, CCI2)
        NOTE_CCI2, nan2 = self.CCI(self.Note, CCI2)
        Gold_CCI2, nan2 = self.CCI(self.Gold, CCI2)

        USD_CCI3, nan3 = self.CCI(self.USD, CCI3)
        VIX_CCI3, nan3 = self.CCI(self.VIX, CCI3)
        NOTE_CCI3, nan3 = self.CCI(self.Note, CCI3)
        Gold_CCI3, nan3 = self.CCI(self.Gold, CCI3)

        USD_CCI4, nan4 = self.CCI(self.USD, CCI4)
        VIX_CCI4, nan4 = self.CCI(self.VIX, CCI4)
        NOTE_CCI4, nan4 = self.CCI(self.Note, CCI4)
        Gold_CCI4, nan4 = self.CCI(self.Gold, CCI4)

        USD_CCI5, nan5 = self.CCI(self.USD, CCI5)
        VIX_CCI5, nan5 = self.CCI(self.VIX, CCI5)
        NOTE_CCI5, nan5 = self.CCI(self.Note, CCI5)
        Gold_CCI5, nan5 = self.CCI(self.Gold, CCI5)

        USD_CCI6, nan6 = self.CCI(self.USD, CCI6)
        VIX_CCI6, nan6 = self.CCI(self.VIX, CCI6)
        NOTE_CCI6, nan6 = self.CCI(self.Note, CCI6)
        Gold_CCI6, nan6 = self.CCI(self.Gold, CCI6)

        USD_CCI7, nan7 = self.CCI(self.USD, CCI7)
        VIX_CCI7, nan7 = self.CCI(self.VIX, CCI7)
        NOTE_CCI7, nan7 = self.CCI(self.Note, CCI7)
        Gold_CCI7, nan7 = self.CCI(self.Gold, CCI7)

        USD_CCI8, nan6 = self.CCI(self.USD, CCI8)
        VIX_CCI8, nan6 = self.CCI(self.VIX, CCI8)
        NOTE_CCI8, nan6 = self.CCI(self.Note, CCI8)
        Gold_CCI8, nan6 = self.CCI(self.Gold, CCI8)

        ########TRI_ 1 계산
        Usd_std = (USD_CCI - self.MA(USD_CCI, CCI1)) / USD_CCI.rolling(CCI1).std()
        Vix_std = (VIX_CCI - self.MA(VIX_CCI, CCI1)) / VIX_CCI.rolling(CCI1).std()

        Final_avg_1 = ((Vix_std * 2) + (Usd_std * 2))/4*(-1)
        TRI_1 = Final_avg_1.rolling(vM1).sum()
        total_nan1 = nan1 + vM1 - 1

        #########TRI_2 계산
        Usd_std2 = (USD_CCI2 - self.MA(USD_CCI2, CCI2)) / USD_CCI2.rolling(CCI2).std()
        Vix_std2 = (VIX_CCI2 - self.MA(USD_CCI2, CCI2)) / VIX_CCI2.rolling(CCI2).std()

        Final_avg_2 = ((Vix_std2 * 2) + (Usd_std2 * 2)) / 4 * (-1)
        TRI_2 = Final_avg_2.rolling(vM2).sum()


        total_nan2 = nan2 + vM2 - 1

        #########TRI_3 계산
        Usd_std3 = (USD_CCI3 - self.MA(USD_CCI3, CCI3)) / USD_CCI3.rolling(CCI3).std()
        Vix_std3 = (VIX_CCI3 - self.MA(VIX_CCI3, CCI3)) / VIX_CCI3.rolling(CCI3).std()

        Final_avg_3 = ((Vix_std3 * 2) + (Usd_std3 * 2))  / 4 * (-1)
        TRI_3 = Final_avg_3.rolling(vM3).sum()
        total_nan3 = nan3 + vM3 - 1



        #########TRI_4 계산
        Usd_std4 = (USD_CCI4 - self.MA(USD_CCI4, CCI4)) / USD_CCI4.rolling(CCI4).std()
        Vix_std4 = (VIX_CCI4 - self.MA(VIX_CCI4, CCI4)) / VIX_CCI4.rolling(CCI4).std()

        Final_avg_4 = ((Vix_std4 * 2) + (Usd_std4 * 2)) / 4 * (-1)
        TRI_4 = Final_avg_4.rolling(vM4).sum()
        total_nan4 = nan4 + vM4 - 1

        #########TRI_5 계산
        Usd_std5 = (USD_CCI5 - self.MA(USD_CCI5, CCI5)) / USD_CCI5.rolling(CCI5).std()
        Vix_std5 = (VIX_CCI5 - self.MA(VIX_CCI5, CCI5)) / VIX_CCI5.rolling(CCI5).std()

        Final_avg_5 = ((Vix_std5 * 2) + (Usd_std5 * 2)) / 4 * (-1)
        TRI_5 = Final_avg_5.rolling(vM5).sum()
        total_nan5 = nan5 + vM5 - 1

        #########TRI_6 계산
        Usd_std6 = (USD_CCI6 - self.MA(USD_CCI6, CCI6)) / USD_CCI6.rolling(CCI6).std()
        Vix_std6 = (VIX_CCI6 - self.MA(VIX_CCI6, CCI6)) / VIX_CCI6.rolling(CCI6).std()

        Final_avg_6 = ((Vix_std6 * 2) + (Usd_std6 * 2)) / 4 * (-1)
        TRI_6 = Final_avg_6.rolling(vM6).sum()
        total_nan6 = nan6 + vM6 - 1


        ########TRI_7 계산
        Usd_std7 = (USD_CCI7 - self.MA(USD_CCI7, CCI7)) / USD_CCI7.rolling(CCI7).std()
        Vix_std7 = (VIX_CCI7 - self.MA(VIX_CCI7, CCI7)) / VIX_CCI7.rolling(CCI7).std()

        Final_avg_7 = ((Vix_std7 * 2) + (Usd_std7 * 2)) / 4 * (-1)
        TRI_7 = Final_avg_7.rolling(vM7).sum()

        #######TRI_8 계산
        Usd_std8 = (USD_CCI8 - self.MA(USD_CCI8, CCI8)) / USD_CCI8.rolling(CCI8).std()
        Vix_std8 = (VIX_CCI8 - self.MA(VIX_CCI8, CCI8)) / VIX_CCI8.rolling(CCI8).std()

        Final_avg_8 = ((Vix_std8 * 2) + (Usd_std8 * 2)) / 4 * (-1)
        TRI_8 = Final_avg_8.rolling(vM8).sum()

        total_nan = 1

        return TRI_1, TRI_2, TRI_3, TRI_4, TRI_5, TRI_6, TRI_7, TRI_8, total_nan

    def TGI(self, vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8):
        HG_CCI, nan1 = self.CCI(self.HG, CCI1)
        HG_CCI2, nan2 = self.CCI(self.HG, CCI2)
        HG_CCI3, nan3 = self.CCI(self.HG, CCI3)
        HG_CCI4, nan4 = self.CCI(self.HG, CCI4)
        HG_CCI5, nan5 = self.CCI(self.HG, CCI5)
        HG_CCI6, nan6 = self.CCI(self.HG, CCI6)
        HG_CCI7, nan7 = self.CCI(self.HG, CCI7)
        HG_CCI8, nan8 = self.CCI(self.HG, CCI8)

        # CCI 값 호출
        data_CCI, nan1 = self.CCI(self.data, CCI1)
        SNP_CCI, nan1 = self.CCI(self.SNP, CCI1)
        DAX_CCI, nan1 = self.CCI(self.DAX, CCI1)
        STOXX_CCI, nan1 = self.CCI(self.STOXX, CCI1)
        Oil_CCI, nan1 = self.CCI(self.Oil, CCI1)
        USD_CCI, nan1 = self.CCI(self.USD, CCI1)

        data_CCI2, nan2 = self.CCI(self.data, CCI2)
        SNP_CCI2, nan2 = self.CCI(self.SNP, CCI2)
        DAX_CCI2, nan2 = self.CCI(self.DAX, CCI2)
        STOXX_CCI2, nan2 = self.CCI(self.STOXX, CCI2)
        Oil_CCI2, nan2 = self.CCI(self.Oil, CCI2)
        USD_CCI2, nan2 = self.CCI(self.USD, CCI2)

        data_CCI3, nan3 = self.CCI(self.data, CCI3)
        SNP_CCI3, nan3 = self.CCI(self.SNP, CCI3)
        DAX_CCI3, nan3 = self.CCI(self.DAX, CCI3)
        STOXX_CCI3, nan3 = self.CCI(self.STOXX, CCI3)
        Oil_CCI3, nan3 = self.CCI(self.Oil, CCI3)
        USD_CCI3, nan3 = self.CCI(self.USD, CCI3)

        data_CCI4, nan4 = self.CCI(self.data, CCI4)
        SNP_CCI4, nan4 = self.CCI(self.SNP, CCI4)
        DAX_CCI4, nan4 = self.CCI(self.DAX, CCI4)
        STOXX_CCI4, nan4 = self.CCI(self.STOXX, CCI4)
        Oil_CCI4, nan4 = self.CCI(self.Oil, CCI4)
        USD_CCI4, nan4 = self.CCI(self.USD, CCI4)

        data_CCI5, nan5 = self.CCI(self.data, CCI5)
        SNP_CCI5, nan5 = self.CCI(self.SNP, CCI5)
        DAX_CCI5, nan5 = self.CCI(self.DAX, CCI5)
        STOXX_CCI5, nan5 = self.CCI(self.STOXX, CCI5)
        Oil_CCI5, nan5 = self.CCI(self.Oil, CCI5)
        USD_CCI5, nan5 = self.CCI(self.USD, CCI5)

        data_CCI6, nan6 = self.CCI(self.data, CCI6)
        SNP_CCI6, nan6 = self.CCI(self.SNP, CCI6)
        DAX_CCI6, nan6 = self.CCI(self.DAX, CCI6)
        STOXX_CCI6, nan6 = self.CCI(self.STOXX, CCI6)
        Oil_CCI6, nan6 = self.CCI(self.Oil, CCI6)
        USD_CCI6, nan6 = self.CCI(self.USD, CCI6)

        data_CCI7, nan7 = self.CCI(self.data, CCI7)
        SNP_CCI7, nan7 = self.CCI(self.SNP, CCI7)
        DAX_CCI7, nan7 = self.CCI(self.DAX, CCI7)
        STOXX_CCI7, nan7 = self.CCI(self.STOXX, CCI7)
        Oil_CCI7, nan7 = self.CCI(self.Oil, CCI7)
        USD_CCI7, nan7 = self.CCI(self.USD, CCI7)

        data_CCI8, nan8 = self.CCI(self.data, CCI8)
        SNP_CCI8, nan8 = self.CCI(self.SNP, CCI8)
        DAX_CCI8, nan8 = self.CCI(self.DAX, CCI8)
        STOXX_CCI8, nan8 = self.CCI(self.STOXX, CCI8)
        Oil_CCI8, nan8 = self.CCI(self.Oil, CCI8)
        USD_CCI8, nan8 = self.CCI(self.USD, CCI8)

        ######TGI_ 1 계산
        NQSNP = data_CCI - SNP_CCI
        DAXSTOXX = DAX_CCI - STOXX_CCI
        OilDollar = Oil_CCI - USD_CCI
        HGCL = HG_CCI-Oil_CCI

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI1)) / NQSNP.rolling(CCI1).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI1)) / DAXSTOXX.rolling(CCI1).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI1)) / OilDollar.rolling(CCI1).std()
        HGCL_std = (HGCL -self.MA(HGCL,CCI1)) / HGCL.rolling(CCI1).std()

        TGI_1 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_1 = TGI_1.rolling(vM1).sum()
        total_nan1 = nan1 + vM1 - 1

        #### TGI_ 2 계산
        NQSNP = data_CCI2 - SNP_CCI2
        SNPDAX = SNP_CCI2 - DAX_CCI2
        DAXSTOXX = DAX_CCI2 - STOXX_CCI2
        OilDollar = Oil_CCI2 - USD_CCI2
        HGCL = HG_CCI2 - Oil_CCI2

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI2)) / NQSNP.rolling(CCI2).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI2)) / SNPDAX.rolling(CCI2).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI2)) / DAXSTOXX.rolling(CCI2).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI2)) / OilDollar.rolling(CCI2).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI2)) / HGCL.rolling(CCI2).std()

        TGI_2 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_2 = TGI_2.rolling(vM2).sum()
        total_nan2 = nan2 + vM2 - 1

        # TGI_3 계산
        NQSNP = data_CCI3 - SNP_CCI3
        SNPDAX = SNP_CCI3 - DAX_CCI3
        DAXSTOXX = DAX_CCI3 - STOXX_CCI3
        OilDollar = Oil_CCI3 - USD_CCI3
        HGCL = HG_CCI3 - Oil_CCI3

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI3)) / NQSNP.rolling(CCI3).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI3)) / SNPDAX.rolling(CCI3).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI3)) / DAXSTOXX.rolling(CCI3).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI3)) / OilDollar.rolling(CCI3).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI3)) / HGCL.rolling(CCI3).std()

        TGI_3 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_3 = TGI_3.rolling(vM3).sum()
        total_nan3 = nan3 + vM3 - 1

        # TGI_4 계산
        NQSNP = data_CCI4 - SNP_CCI4
        SNPDAX = SNP_CCI4 - DAX_CCI4
        DAXSTOXX = DAX_CCI4 - STOXX_CCI4
        OilDollar = Oil_CCI4 - USD_CCI4
        HGCL = HG_CCI4 - Oil_CCI4

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI4)) / NQSNP.rolling(CCI4).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI4)) / SNPDAX.rolling(CCI4).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI4)) / DAXSTOXX.rolling(CCI4).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI4)) / OilDollar.rolling(CCI4).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI4)) / HGCL.rolling(CCI4).std()

        TGI_4 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_4 = TGI_4.rolling(vM4).sum()
        total_nan4 = nan4 + vM4 - 1

        # TGI_5 계산
        NQSNP = data_CCI5 - SNP_CCI5
        SNPDAX = SNP_CCI5 - DAX_CCI5
        DAXSTOXX = DAX_CCI5 - STOXX_CCI5
        OilDollar = Oil_CCI5 - USD_CCI5
        HGCL = HG_CCI5 - Oil_CCI5

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI5)) / NQSNP.rolling(CCI5).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI5)) / SNPDAX.rolling(CCI5).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI5)) / DAXSTOXX.rolling(CCI5).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI5)) / OilDollar.rolling(CCI5).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI5)) / HGCL.rolling(CCI5).std()

        TGI_5 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_5 = TGI_5.rolling(vM5).sum()
        total_nan5 = nan5 + vM5 - 1

        # TGI_6 계산
        NQSNP = data_CCI6 - SNP_CCI6
        SNPDAX = SNP_CCI6 - DAX_CCI6
        DAXSTOXX = DAX_CCI6 - STOXX_CCI6
        OilDollar = Oil_CCI6 - USD_CCI6
        HGCL = HG_CCI6 - Oil_CCI6

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI6)) / NQSNP.rolling(CCI6).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI6)) / SNPDAX.rolling(CCI6).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI6)) / DAXSTOXX.rolling(CCI6).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI6)) / OilDollar.rolling(CCI6).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI6)) / HGCL.rolling(CCI6).std()

        TGI_6 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_6 = TGI_6.rolling(vM6).sum()
        total_nan6 = nan6 + vM6 - 1

        # TGI_7 계산
        NQSNP = data_CCI7 - SNP_CCI7
        SNPDAX = SNP_CCI7 - DAX_CCI7
        DAXSTOXX = DAX_CCI7 - STOXX_CCI7
        OilDollar = Oil_CCI7 - USD_CCI7
        HGCL = HG_CCI7 - Oil_CCI7

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI7)) / NQSNP.rolling(CCI7).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI7)) / SNPDAX.rolling(CCI7).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI7)) / DAXSTOXX.rolling(CCI7).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI7)) / OilDollar.rolling(CCI7).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI7)) / HGCL.rolling(CCI7).std()

        TGI_7 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_7 = TGI_7.rolling(vM7).sum()

        # TGI_8 계산
        NQSNP = data_CCI8 - SNP_CCI8
        SNPDAX = SNP_CCI8 - DAX_CCI8
        DAXSTOXX = DAX_CCI8 - STOXX_CCI8
        OilDollar = Oil_CCI8 - USD_CCI8
        HGCL = HG_CCI8 - Oil_CCI8

        NQSNP_std = (NQSNP - self.MA(NQSNP, CCI8)) / NQSNP.rolling(CCI8).std()
        SNPDAX_std = (SNPDAX - self.MA(SNPDAX, CCI8)) / SNPDAX.rolling(CCI8).std()
        DAXSTOXX_std = (DAXSTOXX - self.MA(DAXSTOXX, CCI8)) / DAXSTOXX.rolling(CCI8).std()
        OilDollar_std = (OilDollar - self.MA(OilDollar, CCI8)) / OilDollar.rolling(CCI8).std()
        HGCL_std = (HGCL - self.MA(HGCL, CCI8)) / HGCL.rolling(CCI8).std()
        TGI_8 = (NQSNP_std + DAXSTOXX_std + OilDollar_std+ HGCL_std) / 4
        TGI_8 = TGI_8.rolling(vM8).sum()

        total_nan = 1

        return TGI_1, TGI_2, TGI_3, TGI_4, TGI_5, TGI_6, TGI_7, TGI_8, total_nan

    def TMI(self, vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8):

        AUD_CCI, nan1 = self.CCI(self.AUD, CCI1)
        CAD_CCI, nan1 = self.CCI(self.CAD, CCI1)
        Peso_CCI, nan1 = self.CCI(self.Peso, CCI1)
        USD_CCI, nan1 = self.CCI(self.USD, CCI1)

        AUD_CCI2, nan2 = self.CCI(self.AUD, CCI2)
        CAD_CCI2, nan2 = self.CCI(self.CAD, CCI2)
        Peso_CCI2, nan2 = self.CCI(self.Peso, CCI2)
        USD_CCI2, nan2 = self.CCI(self.USD, CCI2)

        AUD_CCI3, nan3 = self.CCI(self.AUD, CCI3)
        CAD_CCI3, nan3 = self.CCI(self.CAD, CCI3)
        Peso_CCI3, nan3 = self.CCI(self.Peso, CCI3)
        USD_CCI3, nan3 = self.CCI(self.USD, CCI3)

        AUD_CCI4, nan4 = self.CCI(self.AUD, CCI4)
        CAD_CCI4, nan4 = self.CCI(self.CAD, CCI4)
        Peso_CCI4, nan4 = self.CCI(self.Peso, CCI4)
        USD_CCI4, nan4 = self.CCI(self.USD, CCI4)

        AUD_CCI5, nan5 = self.CCI(self.AUD, CCI5)
        CAD_CCI5, nan5 = self.CCI(self.CAD, CCI5)
        Peso_CCI5, nan5 = self.CCI(self.Peso, CCI5)
        USD_CCI5, nan5 = self.CCI(self.USD, CCI5)

        AUD_CCI6, nan6 = self.CCI(self.AUD, CCI6)
        CAD_CCI6, nan6 = self.CCI(self.CAD, CCI6)
        Peso_CCI6, nan6 = self.CCI(self.Peso, CCI6)
        USD_CCI6, nan6 = self.CCI(self.USD, CCI6)

        AUD_CCI7, nan7 = self.CCI(self.AUD, CCI7)
        CAD_CCI7, nan7 = self.CCI(self.CAD, CCI7)
        Peso_CCI7, nan7 = self.CCI(self.Peso, CCI7)
        USD_CCI7, nan7 = self.CCI(self.USD, CCI7)

        AUD_CCI8, nan8 = self.CCI(self.AUD, CCI8)
        CAD_CCI8, nan8 = self.CCI(self.CAD, CCI8)
        Peso_CCI8, nan8 = self.CCI(self.Peso, CCI8)
        USD_CCI8, nan8 = self.CCI(self.USD, CCI8)

        # TMI 1계산
        AUD_std = (AUD_CCI - self.MA(AUD_CCI, CCI1)) / AUD_CCI.rolling(CCI1).std()
        CAD_std = (CAD_CCI - self.MA(CAD_CCI, CCI1)) / CAD_CCI.rolling(CCI1).std()
        Peso_std = (Peso_CCI - self.MA(Peso_CCI, CCI1)) / Peso_CCI.rolling(CCI1).std()

        TMI_1 = (AUD_std + CAD_std + Peso_std ) / 3
        TMI_1 = TMI_1.rolling(vM1).sum()
        total_nan1 = nan1 + vM1 - 1

        # TMI 2 계산
        AUD_std = (AUD_CCI2 - self.MA(AUD_CCI2, CCI2)) / AUD_CCI2.rolling(CCI2).std()
        CAD_std = (CAD_CCI2 - self.MA(CAD_CCI2, CCI2)) / CAD_CCI2.rolling(CCI2).std()
        Peso_std = (Peso_CCI2 - self.MA(Peso_CCI2, CCI2)) / Peso_CCI2.rolling(CCI2).std()

        TMI_2 = (AUD_std + CAD_std + Peso_std ) / 3
        TMI_2 = TMI_2.rolling(vM2).sum()
        total_nan2 = nan2 + vM1 - 1

        # TMI 3 계산
        AUD_std = (AUD_CCI3 - self.MA(AUD_CCI3, CCI3)) / AUD_CCI3.rolling(CCI3).std()
        CAD_std = (CAD_CCI3 - self.MA(CAD_CCI3, CCI3)) / CAD_CCI3.rolling(CCI3).std()
        Peso_std = (Peso_CCI3 - self.MA(Peso_CCI3, CCI3)) / Peso_CCI3.rolling(CCI3).std()

        TMI_3 = (AUD_std + CAD_std + Peso_std) / 3
        TMI_3 = TMI_3.rolling(vM3).sum()
        total_nan3 = nan3 + vM1 - 1

        # TMI 4 계산
        AUD_std = (AUD_CCI4 - self.MA(AUD_CCI4, CCI4)) / AUD_CCI4.rolling(CCI4).std()
        CAD_std = (CAD_CCI4 - self.MA(CAD_CCI4, CCI4)) / CAD_CCI4.rolling(CCI4).std()
        Peso_std = (Peso_CCI4 - self.MA(Peso_CCI4, CCI4)) / Peso_CCI4.rolling(CCI4).std()

        TMI_4 = (AUD_std + CAD_std + Peso_std) / 3

        TMI_4 = TMI_4.rolling(vM4).sum()
        total_nan4 = nan4 + vM1 - 1

        # TMI 5 계산
        AUD_std = (AUD_CCI5 - self.MA(AUD_CCI5, CCI5)) / AUD_CCI5.rolling(CCI5).std()
        CAD_std = (CAD_CCI5 - self.MA(CAD_CCI5, CCI5)) / CAD_CCI5.rolling(CCI5).std()
        Peso_std = (Peso_CCI5 - self.MA(Peso_CCI5, CCI5)) / Peso_CCI5.rolling(CCI5).std()

        TMI_5 = (AUD_std + CAD_std + Peso_std) / 3
        TMI_5 = TMI_5.rolling(vM5).sum()
        total_nan5 = nan5 + vM1 - 1

        # TMI 6 계산
        AUD_std = (AUD_CCI6 - self.MA(AUD_CCI6, CCI6)) / AUD_CCI6.rolling(CCI6).std()
        CAD_std = (CAD_CCI6 - self.MA(CAD_CCI6, CCI6)) / CAD_CCI6.rolling(CCI6).std()
        Peso_std = (Peso_CCI6 - self.MA(Peso_CCI6, CCI6)) / Peso_CCI6.rolling(CCI6).std()

        TMI_6 = (AUD_std + CAD_std + Peso_std) / 3
        TMI_6 = TMI_6.rolling(vM6).sum()
        total_nan6 = nan6 + vM1 - 1

        # TMI 7 계산
        AUD_std = (AUD_CCI7 - self.MA(AUD_CCI7, CCI7)) / AUD_CCI7.rolling(CCI7).std()
        CAD_std = (CAD_CCI7 - self.MA(CAD_CCI7, CCI7)) / CAD_CCI7.rolling(CCI7).std()
        Peso_std = (Peso_CCI7 - self.MA(Peso_CCI7, CCI7)) / Peso_CCI7.rolling(CCI7).std()

        TMI_7 = (AUD_std + CAD_std + Peso_std) / 3
        TMI_7 = TMI_7.rolling(vM7).sum()

        # TMI 8 계산
        AUD_std = (AUD_CCI8 - self.MA(AUD_CCI8, CCI8)) / AUD_CCI8.rolling(CCI8).std()
        CAD_std = (CAD_CCI8 - self.MA(CAD_CCI8, CCI8)) / CAD_CCI8.rolling(CCI8).std()
        Peso_std = (Peso_CCI8 - self.MA(Peso_CCI8, CCI8)) / Peso_CCI8.rolling(CCI8).std()

        TMI_8 = (AUD_std + CAD_std + Peso_std) / 3
        TMI_8 = TMI_8.rolling(vM8).sum()

        total_nan = 1

        return TMI_1, TMI_2, TMI_3, TMI_4, TMI_5, TMI_6, TMI_7, TMI_8, total_nan

    def STX(self, vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8, STX_A):

        # CCI 계산
        data_CCI, nan1 = self.CCI(self.data, CCI1)
        SNP_CCI, nan1 = self.CCI(self.SNP, CCI1)
        Dow_CCI, nan1 = self.CCI(self.Dow, CCI1)

        data_CCI2, nan2 = self.CCI(self.data, CCI2)
        SNP_CCI2, nan2 = self.CCI(self.SNP, CCI2)
        Dow_CCI2, nan2 = self.CCI(self.Dow, CCI2)

        data_CCI3, nan3 = self.CCI(self.data, CCI3)
        SNP_CCI3, nan3 = self.CCI(self.SNP, CCI3)
        Dow_CCI3, nan3 = self.CCI(self.Dow, CCI3)

        data_CCI4, nan4 = self.CCI(self.data, CCI4)
        SNP_CCI4, nan4 = self.CCI(self.SNP, CCI4)
        Dow_CCI4, nan4 = self.CCI(self.Dow, CCI4)

        data_CCI5, nan5 = self.CCI(self.data, CCI5)
        SNP_CCI5, nan5 = self.CCI(self.SNP, CCI5)
        Dow_CCI5, nan5 = self.CCI(self.Dow, CCI5)

        data_CCI6, nan6 = self.CCI(self.data, CCI6)
        SNP_CCI6, nan6 = self.CCI(self.SNP, CCI6)
        Dow_CCI6, nan6 = self.CCI(self.Dow, CCI6)

        data_CCI7, nan7 = self.CCI(self.data, CCI7)
        SNP_CCI7, nan7 = self.CCI(self.SNP, CCI7)
        Dow_CCI7, nan7 = self.CCI(self.Dow, CCI7)

        data_CCI8, nan8 = self.CCI(self.data, CCI8)
        SNP_CCI8, nan8 = self.CCI(self.SNP, CCI8)
        Dow_CCI8, nan8 = self.CCI(self.Dow, CCI8)

        # STX 1 계산
        data_std = (data_CCI - self.MA(data_CCI, CCI1)) / data_CCI.rolling(CCI1).std()
        SNP_std = (SNP_CCI - self.MA(SNP_CCI, CCI1)) / SNP_CCI.rolling(CCI1).std()
        Dow_std = (Dow_CCI - self.MA(Dow_CCI, CCI1)) / Dow_CCI.rolling(CCI1).std()

        STX_1_1 = data_std.rolling(STX_A).sum()
        STX_1_2 = SNP_std.rolling(STX_A).sum()
        STX_1_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_1_1 + STX_1_2 + STX_1_3) / 3
        STX_1 = avg.rolling(vM1).sum()

        total_nan1 = nan1 + vM1 + STX_A - 2

        # STX 2 계산
        data_std = (data_CCI2 - self.MA(data_CCI2, CCI2)) / data_CCI2.rolling(CCI2).std()
        SNP_std = (SNP_CCI2 - self.MA(SNP_CCI2, CCI2)) / SNP_CCI2.rolling(CCI2).std()
        Dow_std = (Dow_CCI2 - self.MA(Dow_CCI2, CCI2)) / Dow_CCI2.rolling(CCI2).std()

        STX_2_1 = data_std.rolling(STX_A).sum()
        STX_2_2 = SNP_std.rolling(STX_A).sum()
        STX_2_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_2_1 + STX_2_2 + STX_2_3) / 3
        STX_2 = avg.rolling(vM2).sum()
        total_nan2 = nan2 + vM2 + STX_A - 2

        # STX 3 계산
        data_std = (data_CCI3 - self.MA(data_CCI3, CCI3)) / data_CCI3.rolling(CCI3).std()
        SNP_std = (SNP_CCI3 - self.MA(SNP_CCI3, CCI3)) / SNP_CCI3.rolling(CCI3).std()
        Dow_std = (Dow_CCI3 - self.MA(Dow_CCI3, CCI3)) / Dow_CCI3.rolling(CCI3).std()

        STX_3_1 = data_std.rolling(STX_A).sum()
        STX_3_2 = SNP_std.rolling(STX_A).sum()
        STX_3_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_3_1 + STX_3_2 + STX_3_3) / 3
        STX_3 = avg.rolling(vM3).sum()
        total_nan3 = nan3 + vM3 + STX_A - 2

        # STX 4 계산
        data_std = (data_CCI4 - self.MA(data_CCI4, CCI4)) / data_CCI4.rolling(CCI4).std()
        SNP_std = (SNP_CCI4 - self.MA(SNP_CCI4, CCI4)) / SNP_CCI4.rolling(CCI4).std()
        Dow_std = (Dow_CCI4 - self.MA(Dow_CCI4, CCI4)) / Dow_CCI4.rolling(CCI4).std()

        STX_4_1 = data_std.rolling(STX_A).sum()
        STX_4_2 = SNP_std.rolling(STX_A).sum()
        STX_4_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_4_1 + STX_4_2 + STX_4_3) / 3
        STX_4 = avg.rolling(vM4).sum()
        total_nan4 = nan4 + vM4 + STX_A - 2

        # STX 5 계산
        data_std = (data_CCI5 - self.MA(data_CCI5, CCI5)) / data_CCI5.rolling(CCI5).std()
        SNP_std = (SNP_CCI5 - self.MA(SNP_CCI5, CCI5)) / SNP_CCI5.rolling(CCI5).std()
        Dow_std = (Dow_CCI5 - self.MA(Dow_CCI5, CCI5)) / Dow_CCI5.rolling(CCI5).std()

        STX_5_1 = data_std.rolling(STX_A).sum()
        STX_5_2 = SNP_std.rolling(STX_A).sum()
        STX_5_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_5_1 + STX_5_2 + STX_5_3) / 3
        STX_5 = avg.rolling(vM5).sum()
        total_nan5 = nan5 + vM5 + STX_A - 2

        # STX 6 계산
        data_std = (data_CCI6 - self.MA(data_CCI6, CCI6)) / data_CCI6.rolling(CCI6).std()
        SNP_std = (SNP_CCI6 - self.MA(SNP_CCI6, CCI6)) / SNP_CCI6.rolling(CCI6).std()
        Dow_std = (Dow_CCI6 - self.MA(Dow_CCI6, CCI6)) / Dow_CCI6.rolling(CCI6).std()

        STX_6_1 = data_std.rolling(STX_A).sum()
        STX_6_2 = SNP_std.rolling(STX_A).sum()
        STX_6_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_6_1 + STX_6_2 + STX_6_3) / 3
        STX_6 = avg.rolling(vM6).sum()
        total_nan6 = nan6 + vM6 + STX_A - 2

        # STX 7 계산
        data_std = (data_CCI7 - self.MA(data_CCI7, CCI7)) / data_CCI7.rolling(CCI7).std()
        SNP_std = (SNP_CCI7 - self.MA(SNP_CCI7, CCI7)) / SNP_CCI7.rolling(CCI7).std()
        Dow_std = (Dow_CCI7 - self.MA(Dow_CCI7, CCI7)) / Dow_CCI7.rolling(CCI7).std()

        STX_7_1 = data_std.rolling(STX_A).sum()
        STX_7_2 = SNP_std.rolling(STX_A).sum()
        STX_7_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_7_1 + STX_7_2 + STX_7_3) / 3
        STX_7 = avg.rolling(vM7).sum()

        # STX 8 계산
        data_std = (data_CCI8 - self.MA(data_CCI8, CCI8)) / data_CCI8.rolling(CCI8).std()
        SNP_std = (SNP_CCI8 - self.MA(SNP_CCI8, CCI8)) / SNP_CCI8.rolling(CCI8).std()
        Dow_std = (Dow_CCI8 - self.MA(Dow_CCI8, CCI8)) / Dow_CCI8.rolling(CCI8).std()

        STX_8_1 = data_std.rolling(STX_A).sum()
        STX_8_2 = SNP_std.rolling(STX_A).sum()
        STX_8_3 = Dow_std.rolling(STX_A).sum()

        avg = (STX_8_1 + STX_8_2 + STX_8_3) / 3
        STX_8 = avg.rolling(vM8).sum()

        # Trade 계산
        Trade1 = STX_1_1.rolling(vM1).sum()
        Trade2 = STX_2_1.rolling(vM2).sum()
        Trade3 = STX_3_1.rolling(vM3).sum()
        Trade4 = STX_4_1.rolling(vM4).sum()
        Trade5 = STX_5_1.rolling(vM5).sum()
        Trade6 = STX_6_1.rolling(vM6).sum()
        Trade7 = STX_7_1.rolling(vM7).sum()
        Trade8 = STX_8_1.rolling(vM8).sum()

        total_nan = 1

        return STX_1, STX_2, STX_3, STX_4, STX_5, STX_6, STX_7, STX_8, Trade1, Trade2, Trade3, Trade4, Trade5, Trade6, Trade7, Trade8, total_nan

    def fillnan(self, unit):  # unit수만큼 nan값 기입
        data = []
        for step in range(unit):
            data.append(np.NaN)
        return data

    def index_equal(corr, data_date, csv_name_):  # corr값, 날짜

        index = len(corr.dropna())
        corr_ = pd.DataFrame(
            {'corr': corr[-index:].reset_index().drop('index', axis=1).values.reshape(-1)}).dropna().reset_index().drop(
            'index', axis=1)

        date_ = pd.DataFrame({'date': data_date[-index:].reset_index().drop('index', axis=1).values.reshape(
            -1)}).dropna().reset_index().drop('index', axis=1)
        date1 = pd.Series(date_.values.reshape(-1))

        date2 = pd.read_csv(csv_name_)['date']  # 거래내역 데이터의 날짜
        date2 = self.replace_data(date2)

        # 시간 일치하는 날짜,시간 반환
        A = date1.tolist()
        B = date2.tolist()

        C = set(A) & set(B)
        D = [i for i in A if i in B]

        # 일치하는 인덱스 반환
        index_data = []
        for step_date in D:
            ind = date1.tolist().index(step_date)
            index_data.append(ind)

        index_data2 = []
        for step_date in C:
            ind = date2.tolist().index(step_date)
            index_data2.append(ind)

        # 정렬
        index_data.sort()
        index_data2.sort()

        # 일치 및 추출
        date1 = date1[index_data]
        date2 = date2[index_data2]

        corr_ = pd.Series(corr_.values.reshape(-1))
        corr_ = corr_[index_data]
        value = pd.read_csv(csv_name_).iloc[index_data2]

        return date1, date2, corr_, value  # corr 데이터의 날짜, 거래내역 데이터의 날짜, corr



    def NTS_1m(self, vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8, STX_A,
                  TGI_rt, TRI_rt, TMI_rt, STX_rt, Trade_rt, Acc_period, minute):

        self.data_create(minute)
        TRI_1, TRI_2, TRI_3, TRI_4, TRI_5, TRI_6, TRI_7, TRI_8, total_nan_R = self.TRI(vM1, CCI1, vM2, CCI2, vM3, CCI3,
                                                                                       vM4, CCI4, vM5, CCI5, vM6, CCI6,
                                                                                       vM7, CCI7, vM8, CCI8)  # TRI
        TGI_1, TGI_2, TGI_3, TGI_4, TGI_5, TGI_6, TGI_7, TGI_8, total_nan_G = self.TGI(vM1, CCI1, vM2, CCI2, vM3, CCI3,
                                                                                       vM4, CCI4, vM5, CCI5, vM6, CCI6,
                                                                                       vM7, CCI7, vM8, CCI8)  # TGI
        TMI_1, TMI_2, TMI_3, TMI_4, TMI_5, TMI_6, TMI_7, TMI_8, total_nan_M = self.TMI(vM1, CCI1, vM2, CCI2, vM3, CCI3,
                                                                                       vM4, CCI4, vM5, CCI5, vM6, CCI6,
                                                                                       vM7, CCI7, vM8, CCI8)  # TMI
        STX_1, STX_2, STX_3, STX_4, STX_5, STX_6, STX_7, STX_8, Trade1, Trade2, Trade3, Trade4, Trade5, Trade6, Trade7, Trade8, total_nan_S = self.STX(
            vM1, CCI1, vM2, CCI2, vM3, CCI3, vM4, CCI4, vM5, CCI5, vM6, CCI6, vM7, CCI7, vM8, CCI8,
            STX_A)  # STX , Trade
        # nan을 제거하고 인덱스를 통일시킨다
        total_index = min(
            len(TRI_1.dropna()), len(TRI_2.dropna()), len(TRI_3.dropna()), len(TRI_4.dropna()), len(TRI_5.dropna()),
            len(TRI_6.dropna()), len(TRI_7.dropna()), len(TRI_8.dropna()), len(TGI_1.dropna()), len(TGI_2.dropna()),
            len(TGI_3.dropna()), len(TGI_4.dropna()), len(TGI_5.dropna()), len(TGI_6.dropna()), len(TGI_7.dropna()),
            len(TGI_8.dropna()),
            len(TMI_1.dropna()), len(TMI_2.dropna()), len(TMI_3.dropna()), len(TMI_4.dropna()), len(TMI_5.dropna()),
            len(TMI_6.dropna()), len(TMI_7.dropna()), len(TMI_8.dropna()), len(STX_1.dropna()),
            len(STX_2.dropna()), len(STX_3.dropna()), len(STX_4.dropna()), len(STX_5.dropna()), len(STX_6.dropna()),
            len(STX_7.dropna()), len(STX_8.dropna()),
            len(Trade1.dropna()), len(Trade2.dropna()), len(Trade3.dropna()), len(Trade4.dropna()),
            len(Trade5.dropna()), len(Trade6.dropna()), len(Trade7.dropna()), len(Trade8.dropna()))

        TRI_1 = TRI_1[-total_index:].reset_index().drop('index', axis=1)
        TRI_2 = TRI_2[-total_index:].reset_index().drop('index', axis=1)
        TRI_3 = TRI_3[-total_index:].reset_index().drop('index', axis=1)
        TRI_4 = TRI_4[-total_index:].reset_index().drop('index', axis=1)
        TRI_5 = TRI_5[-total_index:].reset_index().drop('index', axis=1)
        TRI_6 = TRI_6[-total_index:].reset_index().drop('index', axis=1)
        TRI_7 = TRI_7[-total_index:].reset_index().drop('index', axis=1)
        TRI_8 = TRI_7[-total_index:].reset_index().drop('index', axis=1)

        TGI_1 = TGI_1[-total_index:].reset_index().drop('index', axis=1)
        TGI_2 = TGI_2[-total_index:].reset_index().drop('index', axis=1)
        TGI_3 = TGI_3[-total_index:].reset_index().drop('index', axis=1)
        TGI_4 = TGI_4[-total_index:].reset_index().drop('index', axis=1)
        TGI_5 = TGI_5[-total_index:].reset_index().drop('index', axis=1)
        TGI_6 = TGI_6[-total_index:].reset_index().drop('index', axis=1)
        TGI_7 = TGI_7[-total_index:].reset_index().drop('index', axis=1)
        TGI_8 = TGI_8[-total_index:].reset_index().drop('index', axis=1)

        TMI_1 = TMI_1[-total_index:].reset_index().drop('index', axis=1)
        TMI_2 = TMI_2[-total_index:].reset_index().drop('index', axis=1)
        TMI_3 = TMI_3[-total_index:].reset_index().drop('index', axis=1)
        TMI_4 = TMI_4[-total_index:].reset_index().drop('index', axis=1)
        TMI_5 = TMI_5[-total_index:].reset_index().drop('index', axis=1)
        TMI_6 = TMI_6[-total_index:].reset_index().drop('index', axis=1)
        TMI_7 = TMI_7[-total_index:].reset_index().drop('index', axis=1)
        TMI_8 = TMI_8[-total_index:].reset_index().drop('index', axis=1)

        STX_1 = STX_1[-total_index:].reset_index().drop('index', axis=1)
        STX_2 = STX_2[-total_index:].reset_index().drop('index', axis=1)
        STX_3 = STX_3[-total_index:].reset_index().drop('index', axis=1)
        STX_4 = STX_4[-total_index:].reset_index().drop('index', axis=1)
        STX_5 = STX_5[-total_index:].reset_index().drop('index', axis=1)
        STX_6 = STX_6[-total_index:].reset_index().drop('index', axis=1)
        STX_7 = STX_7[-total_index:].reset_index().drop('index', axis=1)
        STX_8 = STX_8[-total_index:].reset_index().drop('index', axis=1)

        Trade1 = Trade1[-total_index:].reset_index().drop('index', axis=1)
        Trade2 = Trade2[-total_index:].reset_index().drop('index', axis=1)
        Trade3 = Trade3[-total_index:].reset_index().drop('index', axis=1)
        Trade4 = Trade4[-total_index:].reset_index().drop('index', axis=1)
        Trade5 = Trade5[-total_index:].reset_index().drop('index', axis=1)
        Trade6 = Trade6[-total_index:].reset_index().drop('index', axis=1)
        Trade7 = Trade7[-total_index:].reset_index().drop('index', axis=1)
        Trade8 = Trade8[-total_index:].reset_index().drop('index', axis=1)

        TGRM_1 = (TGI_1 * TGI_rt) + (TRI_1 * TRI_rt) + (TMI_1 * TMI_rt) + (STX_1*STX_rt) + (Trade1*Trade_rt)
        TGRM_1 = TGRM_1 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_2 = (TGI_2 * TGI_rt) + (TRI_2 * TRI_rt) + (TMI_2 * TMI_rt) + (STX_2*STX_rt) + (Trade2*Trade_rt)
        TGRM_2 = TGRM_2 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_3 = (TGI_3 * TGI_rt) + (TRI_3 * TRI_rt) + (TMI_3 * TMI_rt) + (STX_3*STX_rt) +  (Trade3*Trade_rt)
        TGRM_3 = TGRM_3 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_4 = (TGI_4 * TGI_rt) + (TRI_4 * TRI_rt) + (TMI_4 * TMI_rt) + (STX_4*STX_rt) + (Trade4*Trade_rt)
        TGRM_4 = TGRM_4 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_5 = (TGI_5 * TGI_rt) + (TRI_5 * TRI_rt) + (TMI_5 * TMI_rt) + (STX_5*STX_rt) + (Trade5*Trade_rt)
        TGRM_5 = TGRM_5 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_6 = (TGI_6 * TGI_rt) + (TRI_6 * TRI_rt) + (TMI_6 * TMI_rt) + (STX_6*STX_rt) + (Trade6*Trade_rt)
        TGRM_6 = TGRM_6 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_7 = (TGI_7 * TGI_rt) + (TRI_7 * TRI_rt) + (TMI_7 * TMI_rt) + (STX_7*STX_rt) + (Trade7*Trade_rt)
        TGRM_7 = TGRM_7 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM_8 = (TGI_8 * TGI_rt) + (TRI_8 * TRI_rt) + (TMI_8 * TMI_rt) + (STX_8*STX_rt) + (Trade8*Trade_rt)
        TGRM_8 = TGRM_8 / (TGI_rt + TRI_rt + TMI_rt + STX_rt + Trade_rt)

        TGRM = (TGRM_1 + TGRM_2 + TGRM_3 + TGRM_4 + TGRM_5 + TGRM_6 + TGRM_7 + TGRM_8) / 8
        TGRM = TGRM.rolling(Acc_period).sum().dropna().reset_index()[0]
        TGRM = TRI_2

        scaler = MinMaxScaler()  # 0-1사이로 정규화
        scale_TGRM = scaler.fit_transform(TGRM.values.reshape(-1, 1))
        scale_TGRM = pd.Series(scale_TGRM.reshape(-1))
        scale_TGRM = scale_TGRM.fillna(scale_TGRM.mean())


        return TGRM, scale_TGRM


