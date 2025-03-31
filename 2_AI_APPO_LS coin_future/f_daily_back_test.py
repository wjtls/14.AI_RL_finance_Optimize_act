# 일별로 백테스트 하여 플랏
# 일별 손익 표시,

import e_train as params
import a_Env as env_
import c_PPO_Agent as PPO_Agent
import b_network as NET
import torch.multiprocessing as multiprocessing

from dateutil.relativedelta import relativedelta


from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
import psycopg2
import pymysql
import datetime
import yfinance as yf

import e_train as params

env = env_.Env()
    # 데이터 호출

import random
import torch

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Env_:
    def __init__(self, connect, data_name, period, start, end):
        self.conn = connect
        self.data_name = data_name
        self.period = period
        self.start = start
        self.end = end

        self.long_minmax=0
        self.short_minmax=0

        self.agent={}
        self.Global_policy_net={}
        self.Global_value_net={}
        self.agent_data = {}

        self.is_back_testing = True

        # long short 2개 생성
        self.PV_data = {'long': [], 'short': []}
        self.action_data = {'long': [], 'short': []}
        self.buy_data = {'long': [], 'short': []}  # 매수한 가격
        self.sell_data = {'long': [], 'short': []}  # 매도한 가격
        self.buy_date = {'long': [], 'short': []}
        self.sell_date = {'long': [], 'short': []}

        self.date_data = {'long': [], 'short': []}
        self.price_data={'long': [], 'short': []}
        self.scale_input={'long': [], 'short': []}


    def reset(self):
        self.long_ori_data=0
        self.short_ori_data=0

        self.is_back_testing = True

        self.PV_data = {'long': [], 'short': []}
        self.action_data = {'long': [], 'short': []}
        self.buy_data = {'long': [], 'short': []}  # 매수한 가격
        self.sell_data = {'long': [], 'short': []}  # 매도한 가격
        self.buy_date = {'long': [], 'short': []}
        self.sell_date = {'long': [], 'short': []}



    def yf_data_create(self,is_DB_or_yf_data):  # 야후 파이낸스 실시간 데이터 불러오기, NAN 제거
        # 15분으로 설정됨
        if is_DB_or_yf_data=='yf':
            pass
            '''''
            data = yf.download(tickers=self.data_name, period=self.period, start=self.start, end=self.end,
                               interval='15m', group_by='ticker', auto_adjust=True, prepost=True, threads=True,
                               proxy=None, show_errors=True)   # 아예 15분으로 뽑아서 해야됨(시작이 달라지므로)
            res=[]
            minute_=1
            for step in range(round(len(data)/minute_)):
                res.append(data.iloc[step*minute_])
            res_dict={}
            res=pd.DataFrame(res).reset_index()
            res.columns=['Datetime','open','high','low','close','volume']
            res_dict[self.data_name]=res[:-1]  #마지막값은 계속 변동하는 실시간값임
            print('실시간 데이터 호출 완료')
            '''''

        elif is_DB_or_yf_data=='DB':
            db = self.conn.cursor()
            # DB 데이터 호출
            if params.part_backtest==False: # 실시간인경우 전체 호출
                if params.part_time==False:
                    data_count=params.forward_data_count
                    db.execute(f"SELECT symbol,datetime, open, high, low, close, volume FROM (SELECT symbol,datetime,open,high,low,close,volume FROM root.AI_data WHERE symbol={self.data_name} ORDER BY datetime DESC limit {data_count}) as foo order by datetime asc;")
                    data = db.fetchall()

                if params.part_time == True:
                    data_count=params.forward_data_count
                    start_time = params.part_time_data[params.part_time_name][0]
                    end_time = params.part_time_data[params.part_time_name][1]
                    db.execute(
                        f"SELECT symbol,datetime, open, high, low, close, volume FROM (SELECT symbol,datetime, open, high, low, close, volume FROM root.AI_data WHERE symbol= {self.data_name} ORDER BY datetime DESC limit {data_count} ) as foo WHERE extract(hour from datetime) <= {end_time} and extract(hour from datetime) >= {start_time}  order by datetime asc ;")
                    data = db.fetchall()

            else: # price ai의 부분구간을 불러온다
                # offset 이전부터 limit개 까지 불러온다 ex [0] 이전데이터 부터 [1]개 까지
                if params.part_time ==False:
                    db.execute(f"SELECT symbol,datetime, open, high, low, close, volume FROM (SELECT symbol,datetime, open, high, low, close, volume FROM root.AI_data WHERE symbol={params.stock_name} ORDER BY datetime DESC limit {params.test_data_count[1]} offset {params.test_data_count[0]}) as foo order by datetime asc;")
                    data = db.fetchall()

                if params.part_time == True:
                    start_time = params.part_time_data[params.part_time_name][0]
                    end_time = params.part_time_data[params.part_time_name][1]
                    db.execute(
                        f"SELECT symbol,datetime, open, high, low, close, volume FROM (SELECT symbol,datetime, open, high, low, close, volume FROM root.AI_data WHERE symbol= {params.stock_name} ORDER BY datetime DESC limit {params.test_data_count[1]} offset {params.test_data_count[0]}) as foo WHERE extract(hour from datetime) <= {end_time} and extract(hour from datetime) >= {start_time}  order by datetime asc ;")
                    data = db.fetchall()



            minute_ = params.minute

            if len(data) % minute_ == 0:
                res = [data[step * minute_] for step in range(int(np.trunc(len(data) / minute_)))]
            else:
                res = [data[step * minute_] for step in range(int(np.trunc(len(data) / minute_)) + 1)]  # 인터벌 (실시간 전체 다뽑는경우)

            res_dict = {}
            res = pd.DataFrame(res).reset_index()
            res.columns = ['index','symbol','Datetime', 'open', 'high', 'low', 'close', 'volume']

            for name in res.columns:
                if name=='index' or name=='symbol' or name=='Datetime':
                    pass
                else:
                    data_1= [float(value) for value in res[name]] #Decimal 형태를 float으로 변경
                    res[name]=data_1

            res_dict[self.data_name] = res
            return res_dict





    def data_pre2(self, data,is_DB_or_yf_data):  # datetime 한국시각 변경 및 전처리

        data_dict = {}

        if is_DB_or_yf_data == 'yf':  # yf인경우 날짜처리 DB인경우 이미 처리됨
            '''''
            a=[(data[step]['Datetime']=[str(datetime.datetime.strptime(str(data[step]['Datetime'][t])[:-6],'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(hours=13, minutes=10)) for t in range(len(data[step]))] if str(data[step]['Datetime'].iloc[-1])[-6:] == '-04:00' else data[step]['Datetime']=[str(datetime.datetime.strptime(str(data[step]['Datetime'].iloc[t]),'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(hours=13, minutes=10)) for t in range(len(data[step]))] for step in range(len(data[step]))]
            b=[data_dict[self.data_name[step]]=data[step] for step in range(len(data))]
            
            
            '''''
            korea_time = [str(datetime.datetime.strptime(str(data[self.data_name]['Datetime'].iloc[t]),
                                                         '%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=0,
                                                                                                   minutes=0))
                          for t in range(len(data[self.data_name]))]

        else: #DB인경우
            korea_time=[str(data[self.data_name]['Datetime'][t]) for t in range(len(data[self.data_name]))]

        data[self.data_name]['Datetime'] = korea_time
        data_dict[self.data_name] = data[self.data_name]
        close=pd.Series(pd.Series(data_dict[self.data_name]['close']).values*params.point_value)
        open=pd.Series(data_dict[self.data_name]['open'].values*params.point_value)
        high=pd.Series(data_dict[self.data_name]['high'].values*params.point_value)
        low=pd.Series(data_dict[self.data_name]['low'].values*params.point_value)
        vol=pd.Series(data_dict[self.data_name]['volume'].values)
        date=data_dict[self.data_name]['Datetime']
        scaler=MinMaxScaler([0.1,1])
        close_s= scaler.fit_transform(close.values.reshape(-1,1))
        open_s=scaler.fit_transform(open.values.reshape(-1,1))
        high_s= scaler.fit_transform(high.values.reshape(-1,1))
        low_s= scaler.fit_transform(low.values.reshape(-1,1))
        vol_s= scaler.fit_transform(vol.values.reshape(-1,1))

        data_dict= [close,open,high,low,vol,close_s,open_s,high_s,low_s,vol_s,date]


        return data_dict


    def back_test(self,is_short_or_long,long_res,short_res):  # 백테스팅
        #시뮬레이션

        if params.multi_or_thread=='multi':
            self.reset() #시뮬 데이터 리셋
        else: #스레드 학습인경우 data dict 초기화 x (롱과 숏 모두 모아서 계산해야함)
            pass

        self.agent=self.agent_data[is_short_or_long]
        self.agent.reset()  # 리셋 (리셋때 back testing=False 된다)
        self.agent.back_testing = True
        self.is_back_testing=True

        self.agent.scale_input = self.agent.scale_input # 인풋 데이터
        self.agent.price_data = self.agent.price_data  # 종가 데이터

        # 데이터 가공
        self.agent.LSTM_input = self.agent.LSTM_observation(self.agent.scale_input, self.agent.window,
                                                            self.agent.dim)  # LSTM 데이터
        self.agent.LSTM_input_size = self.agent.LSTM_input.size()[2]


        if is_short_or_long == 'short':  # 숏인경우 숏가중치 저장
            policy_net = self.Global_policy_net[is_short_or_long]
            self.decide_action = self.agent.short_decide_action
            self.discrete_step = env.short_discrete_step
            self.Global_lr = params.short_Global_actor_net_lr


        elif is_short_or_long == 'long':  # 롱인경우 롱가중치 저장
            policy_net = self.Global_policy_net[is_short_or_long]
            self.decide_action = self.agent.long_decide_action
            self.discrete_step = env.long_discrete_step
            self.Global_lr = params.long_Global_actor_net_lr


        ##저장된 가중치 load
        policy_net.load()

        # back testing

        for step in range(len(self.agent.price_data)):
            with torch.no_grad():
                prob_ = policy_net(self.agent.LSTM_input).to(self.agent.device)
                policy = F.softmax(prob_,dim=1)  # policy
            self.agent.price = self.agent.price_data[step]  # 현재 주가업데이트
            action, unit = self.decide_action(policy[step],params.deposit)  # 액션 선택
            action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트

            if action == 0: #매도
                self.action_data[is_short_or_long].append(0)
                if unit[0] !=0 : #매매 유닛이 0이 아닌경우
                    self.sell_data[is_short_or_long].append(self.agent.price_data[step])
                    self.sell_date[is_short_or_long].append(step)

            elif action == 1: #관망
                self.action_data[is_short_or_long].append(1)

            else: #매수
                self.action_data[is_short_or_long].append(2)
                if unit[2] != 0:  # 매매 유닛이 0이 아닌경우
                    self.buy_data[is_short_or_long].append(self.agent.price_data[step])
                    self.buy_date[is_short_or_long].append(step)

            # 데이터 저장
            self.PV_data[is_short_or_long].append(self.agent.PV)

            if step % 30 == 0:
                print(step + 1, '/', len(self.agent.price_data), '테스팅중..',float(self.PV_data[is_short_or_long][-1]), ':'+is_short_or_long+'_agent PV' )
                print(policy[step],'step policy')
        long_PV_first = (1 * self.agent.deposit) + (self.agent.price[0] - self.agent.init_cash) * 1 + self.agent.init_cash
        long_PV_last = (1 * self.agent.deposit) + (self.agent.price[-1] - self.agent.init_cash) * 1 + self.agent.init_cash

        self.date_data[is_short_or_long] = self.agent.date_data
        self.price_data[is_short_or_long] = self.agent.price_data
        self.scale_input[is_short_or_long] = self.agent.scale_input
        # 결과
        if params.multi_or_thread=='multi': #멀티프로세싱인경우 Queue에 저장
            if is_short_or_long=='long':
                long_res.put([self])
            else:
                short_res.put([self])
        else:
            if is_short_or_long=='long': #스레드인 경우 리스트 저장
                long_res.append(self)
            else:
                short_res.append(self)

        #print(len(self.PV_data),len(self.action_data),len(self.buy_data),len(self.buy_date),len(self.sell_date),len(self.sell_date),'aksfnkasnfklsnkf')


        return long_res,short_res

    def PV_date_compute2(self,total_date,start_date_Netincome,real_time): #고정값 사용

        '''''
        total_data #data
        start_date_Netincome # 07:00:00
        '''''
        self.compute_size = 0
        daily_index_zip=[]
        total_date_data = total_date[params.window-1:] #윈도우사이즈만큼 제거

        for step in range(len(total_date_data)):
            if total_date_data[step][11:13]==start_date_Netincome[:2]:
                daily_index_zip.append(step)

        daily_index_zip2=[0]
        for step in range(len(daily_index_zip)-1):
            if total_date_data[daily_index_zip[step]][8:10] != total_date_data[daily_index_zip[step+1]][8:10]:
                daily_index_zip2.append(daily_index_zip[step])

            if len(daily_index_zip)-2==step: #last step
                daily_index_zip2.append(daily_index_zip[step+1])

        day_index=daily_index_zip2[-1]
        week_index=0
        weekly_index_zip=0

        return day_index, week_index, daily_index_zip2, weekly_index_zip

    def PV_date_compute(self,total_date,start_date_Netincome,real_time):  # 당일 시간 인덱스, 일주일전 월요일 7시의 인덱스 반환
        self.compute_size=0
        total_date_data = total_date[-(self.compute_size - params.window + 1):]
        now_=datetime.datetime.now()
        now = str(now_) # 현재 날짜 ex 2022-07-22

        before_one_day= str(now_-relativedelta(days=1))
        before_one_week_ = now_ - relativedelta(weeks=1)
        before_one_week = str(now_ - relativedelta(weeks=1))[:10] #1주일전 날짜 반환 ex 2022-07-15
        if before_one_week == 5 or before_one_week == 6 :#1주전이 주말인경우 -2일 추가  ex 2022-07-13
            before_one_week= str(before_one_week_ - relativedelta(days=2))[:10]

        start_data1 = [now[:10] + ' ' + start_date_Netincome[:4] + str_int + start_date_Netincome[5:] for str_int in [str(k) for k in range(10)]] #오늘의 년월일 + 기준 시간
        start_data1_ = [now[:10] + ' ' + start_date_Netincome[:3] + str_int + start_date_Netincome[5:] for str_int in [str(k) for k in range(10,59)]] #오늘의 년월일 + 기준 시간
        start_data1=start_data1+start_data1_

        start_data2=[before_one_day[:10] + ' ' + start_date_Netincome[:4] + str_int + start_date_Netincome[5:]  for str_int in [str(k) for k in range(10)]]
        start_data2_=[before_one_day[:10] + ' ' + start_date_Netincome[:3] + str_int + start_date_Netincome[5:]  for str_int in [str(k) for k in range(10,59)]]
        start_data2=start_data2+start_data2_

        start_data= np.array([start_data1,start_data2]).flatten()

        for start_date in start_data:
            day = [-(len(total_date_data)-total_date_data.index(k)) for k in total_date_data if str(k)==str(start_date)]
            if day != []: #값이 있으면 멈춤
                break


        day_index=int(day[0])
        day_index_copy=day_index
        week_index=0

        daily_index_zip=[day_index]
        weekly_index_zip=0

        daily_period=round(1380/params.minute)  #하루 1440 분 / 기준 분

        while True:
            day_index_copy -= daily_period
            if -(len(total_date)-(params.window-1))> day_index_copy:
                daily_index_zip.append(-int(len(total_date)-(params.window-1)))
                break
            else:
                daily_index_zip.append(int(day_index_copy))


        return day_index,week_index,daily_index_zip,weekly_index_zip




    def forward(self): #전진분석
        #DB데이터(학습했던 구간)의 min값, max값과 실시간 데이터를 합치고 전진분석
        #처음 한번 불러와서 ori 지표값 max min 구한다음에 실시간데이터 지표값에 넣고 minmaxscaler
        # 학습시 뽑았던 지표 ori data 처음 한번만 뽑음

            # data min과 max를 실시간에 추가해야한다 (실시간 데이터 뽑아서 스케일링시 값이 변하는 문제 없에야함)

        pass

#####csv및 text데이터 업데이트

#####야후 파이낸스 데이터 업데이트###dbx너무
# 특정기간 None있으면 오류 날거 예상































if __name__ == '__main__':
    period = '5d'
    start = None
    end = None
    connect = pymysql.connect(host='192.168.35.22', user='root', password='Gywk102011!', db='root',
                                  charset='utf8',
                                  port=3306)

    Future_name=params.stock_name_ #price_ai2 이름
    is_DB_or_yf_data='DB'

    #####################학습시 지표 데이터

    bk = Env_(connect, Future_name, period, start, end)

    # past_data_ = env.data_create(params.minute, params.data_count, params.coin_or_stock, params.point_value)
    past_data_ = env.coin_data_create(params.minute, params.data_count, params.coin_or_stock, params.point_value) #API에서 학습했던 부분 바로 불러옴( DB안거침)
    data_count= len(past_data_[0]) #가져온 데이터길이만큼
    long_input_, short_input_, ori_data = env.input_create(params.minute, params.ratio, data_count,
                                                           params.coin_or_stock, params.point_value,
                                                           params.short_ind_name, params.long_ind_name,
                                                           past_data_)  # ind에서 높은 값을 뽑음
    idx = 0

    long_train_data, long_val_data, long_test_data, long_ori_close, long_total_input, long_date_data, long_total_date = long_input_\


    while True:
        fig = plt.figure(num=1,figsize=(10, 10))  # figure(도표) 생성
        g1 = fig.add_subplot(511)
        g2 = fig.add_subplot(512)
        g3 = fig.add_subplot(513)
        g4 = fig.add_subplot(514)
        g5= fig.add_subplot(515)
        #############################백테스팅시 지표###############
        real_insert.forward()
        real_data = env.coin_data_create(params.minute, params.data_count, params.coin_or_stock, params.point_value) #API에서 학습했던 부분 바로 불러옴( DB안거침) # 실시간 데이터 불러옴 (기준시각: 표준시각)
        real_data_dict = bk.data_pre2(real_data,is_DB_or_yf_data) #불러온 데이터 추가 처리 (한국시간으로 변경가능)
        print(real_data,'real_aatasf')
        data_count=len(real_data) #인덱싱할 데이터수
        # 실시간 데이터
        long_input, short_input, real_ori_data = env.input_create(params.minute, params.ratio, data_count,
                                                                  params.coin_or_stock, params.point_value,
                                                                  params.short_ind_name, params.long_ind_name,
                                                                  real_data_dict)  # ind에서 높은 값을 뽑음

        long_train_data, long_val_data, long_test_data, long_ori_close, long_total_input, long_date_data, long_total_date = long_input
        short_train_data, short_val_data, short_test_data, short_ori_close, short_total_input, short_date_data, short_total_date = short_input

        bk.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
        bk.long_scale_input = long_total_input
        bk.long_date_data = long_total_date

        bk.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
        bk.short_scale_input = short_total_input
        bk.short_date_data = short_total_date

        bk.long_ori_data = ori_data[0]  # 학습때 썼던 ori data 저장
        bk.short_ori_data = ori_data[1]
        long_real_ori = real_ori_data[0]
        short_real_ori = real_ori_data[1]


        #리 페인트 방지위해 최대 최소 값 넣고 스케일링

        long_maxmin = [[np.max(long_data), np.min(long_data)] for long_data in bk.long_ori_data]
        short_maxmin = [[np.max(short_data), np.min(short_data)] for short_data in bk.short_ori_data]


        long_res_data = []
        short_res_data = []

        for index, max_min in enumerate(long_maxmin):  # 과거의 최대, 최소값 real에 추가
            res_data = []
            for step in range(len(long_real_ori[index])):
                try:
                    res = np.array(long_real_ori[index].iloc[step])
                except:
                    res = np.array(long_real_ori[index][step])

                if max_min[index] < res and index == 0:  # 인덱스0(롱에서) 최대값보다 ori값이 크면 최대값으로 고정
                    res = max_min[index]
                if max_min[index] > res and index == 1:  # 인덱스1(숏에서) 최소값보다 ori값이 작으면 최소값으로 고정
                    res = max_min[index]
                res = np.insert(res, 0, np.array(max_min))

                scaler = MinMaxScaler([0.1, 1])
                res = scaler.fit_transform(res.reshape(-1, 1))
                res_data.append(res[-1])
            long_res_data.append(torch.Tensor([res_data]).view(-1))

        for index, max_min in enumerate(short_maxmin):  # 과거의 최대, 최소값 real에 추가
            res_data = []
            for step in range(len(short_real_ori[index])):
                try:
                    res = np.array(short_real_ori[index].iloc[step])
                except:
                    res = np.array(short_real_ori[index][step])

                if max_min[index] < res and index == 0:  # 인덱스0(롱에서) 최대값보다 ori값이 크면 최대값으로 고정
                    res = max_min[index]
                if max_min[index] > res and index == 1:  # 인덱스1(숏에서) 최소값보다 ori값이 작으면 최소값으로 고정
                    res = max_min[index]

                res = np.insert(res, 0, np.array(max_min))
                scaler = MinMaxScaler([0.1, 1])
                res = scaler.fit_transform(res.reshape(-1, 1))
                res_data.append(res[-1])
            short_res_data.append(torch.Tensor([res_data]).view(-1))
        '''''
        long_res_data = []
        short_res_data = []

        for index, max_min in enumerate(long_maxmin):  # 과거의 최대, 최소값 real에 추가
            res = np.insert(long_real_ori[index], 0, np.array(max_min))
            scaler = MinMaxScaler([0.1, 1])
            res = scaler.fit_transform(res.reshape(-1, 1))
            long_res_data.append(torch.Tensor([res[2:]]).view(-1))


        for index, max_min in enumerate(short_maxmin):
            res = np.insert(short_real_ori[index], 0, max_min)
            scaler = MinMaxScaler([0.1, 1])
            res = scaler.fit_transform(res.reshape(-1, 1))
            short_res_data.append(torch.Tensor([res[2:]]).view(-1))
        '''''

        # 에이전트 호출 및 처리한 데이터 에이전트에 넣기
        for short_or_long in params.short_or_long_data:
            # global net
            Global_actor = NET.Global_actor
            Global_critic = NET.Global_critic
            bk.Global_policy_net[short_or_long] = Global_actor(params.device, params.window,
                                                                 short_or_long)  # dict 에 글로벌넷 저장
            bk.Global_value_net[short_or_long] = Global_critic(params.device, params.window, short_or_long)

            # 숏 or 롱 포지션 따라 인풋 정의
            if short_or_long == 'short':
                #input_ = bk.short_scale_input
                input_ = short_res_data
                bk.input_dim = params.input_dim['short']
                ori_close = bk.short_price_data
                date_data = bk.short_date_data

            else:
                input_ = long_res_data
                bk.input_dim = params.input_dim['long']
                ori_close = bk.long_price_data
                date_data = bk.long_date_data


            agent_num = 0  # 글로벌 에이전트 넘버=0

            # 에이전트 정의  ( bk의 init에서 agent 를 넣는데 한번더 덮어씌움)
            bk.agent_data[short_or_long] = PPO_Agent.PPO(params.window,  # LSTM 윈도우 사이즈
                                                           params.cash,  # 초기 보유현금
                                                           params.cost,  # 수수료 %
                                                           params.device,  # 디바이스 cpu or gpu
                                                           params.k_epoch,  # K번 반복
                                                           input_,  # 인풋 데이터
                                                           ori_close,  # 주가 데이터
                                                           date_data,  # 날짜 데이터
                                                           bk.input_dim,  # feature 수
                                                           agent_num,
                                                           params.coin_or_stock,
                                                           params.deposit,
                                                           params.backtest_slippage,
                                                           short_or_long,  # 숏인지 롱인지
                                                           bk.Global_policy_net[short_or_long],  # 글로벌넷
                                                           bk.Global_value_net[short_or_long]  # 글로벌넷
                                                           )



        # 실시간 전진 분석
        process_list = []
        res_data = {}
        long_res = multiprocessing.Queue()  # 결과 저장
        short_res = multiprocessing.Queue()

        print('병렬 백테스트 시작.', '코어수:', multiprocessing.cpu_count())

        if params.backtest_hedge_on_off == 'on':
            if params.multi_or_thread == 'multi':  # 멀티 프로세싱인경우 (메모리 많이잡아먹음)
                if __name__ == '__main__':
                    for is_short_or_long in params.short_or_long_data:  # func 목록 모두 저장
                        proc = multiprocessing.Process(target=bk.back_test,
                                                       args=([is_short_or_long, long_res, short_res]))
                        process_list.append(proc)
                    [process.start() for process in process_list]

                    for step in range(len(params.short_or_long_data)):  # 숏, 롱 데이터 큐에 저장
                        is_short_or_long = params.short_or_long_data[step]
                        if is_short_or_long == 'long':
                            res_ = long_res.get()[0]
                        if is_short_or_long == 'short':
                            res_ = short_res.get()[0]
                        res_data[is_short_or_long] = res_

                    [process.join() for process in process_list]  # 종료

            else:  # 멀티프로세싱 아닌경우
                long_res_ = []
                short_res_ = []
                for is_short_or_long in params.short_or_long_data:  # 스레드 시뮬레이션 ( 메모리 낮은경우 대비)
                    long_res, short_res = bk.back_test(is_short_or_long, long_res_, short_res_)
                    if is_short_or_long == 'long':
                        res_ = long_res[0]
                    if is_short_or_long == 'short':
                        res_ = short_res[0]
                    res_data[is_short_or_long] = res_

        if __name__ == '__main__': ######플랏
            if bk.is_back_testing == True:  # 백테스팅일경우 출력

                total_dim = len(params.short_or_long_data)


                #지표 길이 다를 경우 인덱스 값 맞춘다
                ind_diff = {}
                if len(res_data['long'].PV_data['long']) > len(res_data['long'].PV_data['short']):  # 롱데이터수 더길면 (롱의 지표변수가 더짧다)

                    ind_diff['long'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))  # 그래프 계산시 얼마나 빼야할지
                    ind_diff['short'] = 0

                    PV_data = res_data['long'].PV_data['long']
                    res_data['long'].PV_data['long'] = PV_data[ind_diff['long']:]

                    #매매date 인덱스 조절
                    res_data['long'].buy_date['long'] = res_data['long'].buy_date['long'] - ind_diff['long'] # 인덱스 조절
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'] - ind_diff['long']  # 인덱스 조절

                    #사고 판 가격 인덱스 조절
                    res_data['long'].buy_data['long'] = res_data['long'].buy_data['long'][len(res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] < 0]):]
                    res_data['long'].sell_data['long'] = res_data['long'].sell_data['long'][len(res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] < 0]):]

                    # 매매 date 인덱스 조절2
                    res_data['long'].buy_date['long'] = res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] >= 0].tolist()
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] >= 0].tolist()

                    # 가격, 날짜 인덱스 조절
                    res_data['long'].agent.price_data = bk.price_data['long'][ind_diff['long']:]
                    res_data['long'].agent.date_data = bk.date_data['long'][ind_diff['long']:]


                elif len(res_data['long'].PV_data['long']) < len(res_data['long'].PV_data['short']):  # 숏이20 더길면 숏에 20이 들어감
                    ind_diff['short'] = np.abs(len(res_data['short'].PV_data['short']) - len(res_data['long'].PV_data['long']))
                    ind_diff['long'] = 0

                    PV_data = res_data['short'].PV_data['short']
                    res_data['short'].PV_data['short'] = PV_data[ind_diff['short']:]

                    # 매매date 인덱스 조절
                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'] - ind_diff['short']  # 인덱스 조절
                    res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'] - ind_diff['short']  # 인덱스 조절

                    # 사고 판 가격 인덱스 조절
                    res_data['short'].buy_data['short'] = res_data['short'].buy_data['short'][len(
                        res_data['short'].buy_date['short'][res_data['short'].buy_date['short'] < 0]):]
                    res_data['short'].sell_data['short'] = res_data['short'].sell_data['short'][len(
                        res_data['short'].sell_date['short'][res_data['short'].sell_date['short'] < 0]):]

                    # 매매 date 인덱스 조절2
                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'][
                        res_data['short'].buy_date['short'] >= 0].tolist()
                    res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'][
                        res_data['short'].sell_date['short'] >= 0].tolist()

                    # 가격, 날짜 인덱스 조절
                    res_data['short'].agent.price_data = bk.price_data['short'][ind_diff['short']:]
                    res_data['short'].agent.date_data = bk.date_data['short'][ind_diff['short']:]



                else:  # 길이같으면
                    ind_diff['short'] = 0
                    ind_diff['long'] = 0


                for dim in range(total_dim):
                    is_short_or_long = params.short_or_long_data[dim]
                    # 앞에 self 붙으면 class안에 중복된 이름있기 때문에 dict이 소실됨
                    agent = res_data[is_short_or_long].agent
                    PV_data = res_data[is_short_or_long].PV_data[is_short_or_long]
                    action_data = res_data[is_short_or_long].action_data[is_short_or_long]
                    buy_data = res_data[is_short_or_long].buy_data[is_short_or_long]
                    buy_date = res_data[is_short_or_long].buy_date[is_short_or_long]
                    sell_data = res_data[is_short_or_long].sell_data[is_short_or_long]
                    sell_date = res_data[is_short_or_long].sell_date[is_short_or_long]
                    price_data = res_data[is_short_or_long].agent.price_data.view(-1)

                    total_date = 0
                    if is_short_or_long == 'long':
                        total_date = long_total_date
                    else:
                        total_date = short_total_date

                    day_index, week_index, daily_index_zip, weekly_index_zip = bk.PV_date_compute2(total_date,
                                                                                                    '18:00:00', 'False') #미국 시간


                    day_index= day_index - (ind_diff['short']+ ind_diff['long'])
                    week_index= week_index - (ind_diff['short']+ ind_diff['long'])
                    daily_index_zip = daily_index_zip - (ind_diff['short']+ ind_diff['long'])
                    weekly_index_zip = weekly_index_zip - (ind_diff['short']+ ind_diff['long'])



                    #PV를 뽑아서 계산한다.

                    init_PV = params.cash  # PV

                    if dim == 0:  # 처음 dim에 출력
                        if len(params.short_or_long_data) > 1:  # 헷지모드 on( 롱숏 둘다 백테스트 하며 각각 PV연산하여 합산)
                            long_agent = res_data['long'].agent
                            long_PV_data = res_data['long'].PV_data['long']

                            short_agent = res_data['short'].agent
                            short_PV_data = res_data['short'].PV_data['short']

                            long_data_date = long_agent.date_data
                            short_data_date = short_agent.date_data

                            # 길이 일치
                            PV_data_set = [long_PV_data, short_PV_data]
                            date_set = [long_data_date, short_data_date]

                            less_data = PV_data_set[np.argmin([len(long_PV_data), len(short_PV_data)])]  # 갯수가 더 적은 데이터
                            more_data = PV_data_set[np.argmax([len(long_PV_data), len(short_PV_data)])]
                            more_date = date_set[np.argmax([len(long_data_date), len(short_data_date)])]  # 갯수 더 많은 날짜 데이터
                            len_diff = np.abs(len(long_PV_data) - len(short_PV_data))  # 차이


                            if less_data==PV_data_set[0]: #데이터수 적은 에이전트
                                less_agent=long_agent
                                len_less_data=len(less_data)
                            else:
                                less_agent=short_agent
                                len_less_data=len(less_data)

                            less_data = torch.cat([torch.ones(len_diff)*(less_agent.init_cash), torch.Tensor(less_data).view(-1)]).view(-1)  # 적었던 PV데이터 0으로 채워서 길이 일치

                            if len_diff == 0:  # 차이가 없는경우
                                more_data = long_PV_data
                                less_data = short_PV_data

                            if less_agent==short_agent: # 데이터수 적은게 숏인경우
                                short_res_PV=torch.Tensor(more_data[-len_less_data:]).view(-1) + torch.Tensor(less_data[-len_less_data:]).view(-1)
                                long_res_PV= torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)
                            elif less_agent==long_agent: # 데이터수 적은게 롱인경우
                                short_res_PV=torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)
                                long_res_PV=torch.Tensor(more_data[-len_less_data:]).view(-1) + torch.Tensor(less_data[-len_less_data:]).view(-1)

                            res_PV = torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)  # PV 합
                            res_date = more_date


                            g1.set_ylabel('NI of short and long Agent')
                            g1.plot(res_date, res_PV[-len(res_date):] - (long_agent.init_cash + short_agent.init_cash))

                        elif len(params.short_or_long_data) == 1:  # short or long 인경우
                            g1.set_ylabel('NI_of_' + is_short_or_long + '_Agent')
                            g1.plot(agent.date_data, torch.Tensor(PV_data) - agent.init_cash)


                    if is_short_or_long == 'long':  # 롱인경우

                        plot_size = 500
                        sell_plot_size = sell_date.index(np.array(sell_date)[np.array(sell_date) > len(long_res_PV) - plot_size][0])
                        diff_sell_value = len(long_res_PV)-plot_size

                        buy_plot_size = buy_date.index(np.array(buy_date)[np.array(buy_date) > len(long_res_PV) - plot_size][0])
                        diff_buy_value = len(long_res_PV)-plot_size

                        g2.set_ylabel(is_short_or_long + '_AI')
                        g2.scatter(buy_date, buy_data, marker='v', color='red')
                        g2.scatter(sell_date, sell_data, marker='v', color='blue')
                        g2.plot(price_data)


                        g4.set_ylabel(is_short_or_long + '_AI')
                        g4.scatter(np.array(buy_date[buy_plot_size:])-diff_buy_value, buy_data[buy_plot_size:], marker='v', color='red')
                        g4.scatter(np.array(sell_date[sell_plot_size:])-diff_sell_value, sell_data[sell_plot_size:], marker='v', color='blue')
                        g4.plot(price_data[-plot_size:])


                    if is_short_or_long == 'short':  # 숏인경우
                        try:
                            plot_size = 500
                            sell_plot_size = sell_date.index(np.array(sell_date)[np.array(sell_date) > len(short_res_PV) - plot_size][0])
                            diff_sell_value = len(short_res_PV)-plot_size


                            buy_plot_size = buy_date.index(np.array(buy_date)[np.array(buy_date) > len(short_res_PV) - plot_size][0])
                            diff_buy_value = len(short_res_PV)-plot_size


                            g3.set_ylabel(is_short_or_long + '_AI')
                            g3.scatter(buy_date, buy_data, marker='v', color='blue')
                            g3.scatter(sell_date, sell_data, marker='v', color='red')
                            g3.plot(price_data)

                            g5.set_ylabel(is_short_or_long + '_AI')
                            g5.scatter(np.array(buy_date[buy_plot_size:])-diff_buy_value, buy_data[buy_plot_size:], marker='v', color='blue')
                            g5.scatter(np.array(sell_date[sell_plot_size:])-diff_sell_value, sell_data[sell_plot_size:], marker='v', color='red')
                            g5.plot(price_data[-plot_size:])

                            plt.text(0.7, 0.7, 'Type E,  Long:up,down>LRCCI LRLRA up down    short:up,down,LRLRA,LRCCI')
                            plt.text(0.7, 0.2, 'Long Net income: ' + str(long_PV_data[-1] - long_PV_data[day_index]))
                            plt.text(0.7, 0.1, 'Short Net income: ' + str(short_PV_data[-1] - short_PV_data[day_index]))
                            plt.text(0.7, 0.6, 'daily Net income: ' + str(res_PV[-1] - res_PV[day_index]))
                        except:
                            pass

                print('APPO_LS total NI :', res_PV[-1] - (long_agent.init_cash + short_agent.init_cash))
                print('APPO_LS Daily Net income :', res_PV[-1] - res_PV[day_index])


                print('---------------------------------------------------------------------------------')
                daily_date=[]
                daily_total_NI=[]
                daily_short_NI=[]
                daily_long_NI=[]

                total_date=total_date[params.window-1:]
                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily 순 수익금 :',res_PV[daily_index_zip[step+1]]-res_PV[daily_index_zip[step]])
                    daily_date.append(total_date[daily_index_zip[step+1]])
                    daily_total_NI.append((res_PV[daily_index_zip[step+1]]-res_PV[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')

                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily long 순 수익금 : ', long_PV_data[daily_index_zip[step+1]]-long_PV_data[daily_index_zip[step]])
                    daily_long_NI.append((long_PV_data[daily_index_zip[step+1]]-long_PV_data[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')



                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily short 순 수익금 :', PV_data[daily_index_zip[step+1]]-PV_data[daily_index_zip[step]])
                    daily_short_NI.append((PV_data[daily_index_zip[step+1]]-PV_data[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')


                all_date = pd.Series(total_date[params.window-1:])
                all_PV= pd.Series(res_PV)
                all_PV=pd.concat([all_date,all_PV],axis=1)
                all_PV.to_csv('all_PV')

                daily_date=pd.Series(daily_date)
                daily_total_NI=pd.Series(daily_total_NI)
                daily_short_NI=pd.Series(daily_short_NI)
                daily_long_NI=pd.Series(daily_long_NI)

                daily_total_NI=pd.concat([daily_date,daily_total_NI],axis=1)
                daily_short_NI=pd.concat([daily_date,daily_short_NI],axis=1)
                daily_long_NI=pd.concat([daily_date,daily_long_NI],axis=1)

                daily_total_NI.columns=['date','NI']
                daily_short_NI.columns=['date','NI']
                daily_long_NI.columns=['date','NI']

                daily_total_NI.to_csv('za_realtime_total_NI')
                daily_short_NI.to_csv('za_realtime_short_NI')
                daily_long_NI.to_csv('za_realtime_long_NI')


                print('사용지표 Long:지지 저항    short:지지저항 학습 이후 LRCCI LRLRA 지지 저항')




            plt.show(block=False)
            plt.pause(300)
            plt.close()
            bk.reset()



