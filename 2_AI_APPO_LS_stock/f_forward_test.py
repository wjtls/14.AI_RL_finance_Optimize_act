import pyupbit as py
import numpy as np
import pandas as pd
import requests
import e_train as params
import a_Env as env_
import ccxt
import torch
import b_network as NET
import c_PPO_Agent as PPO_Agent
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import random
import torch
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd
import math
from a_korea_invest_api_env import get_ovsstk_chart_price as get_price
import mojito
import pprint
import json

seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if params.device=='cuda':
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


env=env_.Env()
params.real_or_train='real'

if params.trading_site =='binance':
    API_key =params.A
    Secret_key =params.stock_Secret_key

if params.trading_site =='upbit':
    API_key = params.Upbit_API_key
    Secret_key = params.Upbit_Secret_key


def real_save_price_data(data,ind_data,data_name):
    data_name = data_name.replace('/', '_')
    data = [pd.Series(data[step],name='data').reset_index()['data'] for step in range(len(data))]

    data_ = pd.DataFrame(data[:-1])
    date = pd.DataFrame(data[-1])
    data_minute = pd.DataFrame([params.minute])

    date_name_ = str('save_real_date_'+data_name)
    data_name_ = str('save_real_price_'+data_name)
    minute_name = 'minute_data'

    date.to_csv(date_name_,index=False)
    data_.to_csv(data_name_,index=False)
    torch.save(ind_data,'save_real_ind_'+data_name)
    data_minute.to_csv(minute_name,index=False)




def real_load_price_data(data_name,real_data_count):  #불러온 데이터를 csv로 저장하고 동일한 날짜인경우 불러올때 csv를 호출함으로써 시간 절약
    res = 0
    data = 0
    data_name = data_name.replace('/', '_')
    ind_data_ = [0, 0, 0]

    try:
        csv_data=pd.read_csv('save_real_date_'+data_name).values
        ind_data = torch.load('save_real_ind_'+data_name)
        past_minute =pd.read_csv('minute_data').values[0][0] #과거 분봉

        if params.minute == past_minute and real_data_count[1] == csv_data[-1][0][:16]:
            res='csv'
            csv_data_ = pd.read_csv('save_real_price_'+data_name).values
            data = [pd.Series(csv_data_[step]) for step in range(len(csv_data_))]
            data.append(pd.Series(csv_data.reshape(-1))) # 날짜 추가
            ind_data_ = [pd.Series(ind_data[step]) for step in range(len(ind_data))]
        else :
            #API와 저장된 데이터의 불러올 날짜가 다르면 새로 API 호출
            print('불러온 데이터의 마지막 날짜:',csv_data[-1][0][:16],'         불러온 최근 시간:',real_data_count[1])
            res='API'
            csv_data_ = pd.read_csv('save_real_price_' + data_name).values
            data = [pd.Series(csv_data_[step]) for step in range(len(csv_data_))]
            data.append(pd.Series(csv_data.reshape(-1)))  # 날짜 추가
            ind_data_ = [pd.Series(ind_data[step]) for step in range(len(ind_data))]


    except Exception as e:  # 예외 유형을 Exception으로 지정
        print('저장된 데이터 파일이 없습니다. 새로운 API 호출 실시')
        print(f'오류 메시지: {e}')  # 오류 메시지 출력
        res = 'API'


    return res,data,ind_data_







####################코인 트레이딩 일봉단위 의사결정
class stock_Env():
    def __init__(self):

        self.Global_policy_net={}
        self.Global_value_net={}
        self.agent_data={}
        self.agent=0
        self.decide_action=0


        self.myToken = params.Slack_token

        self.API_key = params.stock_API_key
        self.Secret_key= params.stock_Secret_key


        self.broker = mojito.KoreaInvestment(
            api_key=params.stock_API_key,
            api_secret=params.stock_Secret_key,
            acc_no=params.account_number,
            exchange='나스닥'
        )

        self.is_back_testing = True
        self.symbol_name = params.API_stock_name
        self.ori_symbol_name = params.API_stock_name


        # long short 2개 생성
        self.PV_data = {'long': [], 'short': []}
        self.action_data = {'long': [], 'short': []}
        self.buy_data = {'long': [], 'short': []}  # 매수한 가격
        self.sell_data = {'long': [], 'short': []}  # 매도한 가격
        self.buy_date = {'long': [], 'short': []}
        self.sell_date = {'long': [], 'short': []}
        self.price_data = {'long': [], 'short': []}  # 가격 데이터
        self.date_data = {'long': [], 'short': []}  # 날짜 데이터
        self.scale_input = {'long': [], 'short': []}

        self.main_ind_state = 'init'
        self.agent_data = {}
        self.past_data_date = 0 #불러올 날짜 비교
        self.past_ind_data = 0 #리페인트 비교

        self.access_token, self.access_token_expired = get_price.get_access_token()

    def round_down_num(self, number):
        res = math.floor(number)  # 주어진 숫자 반내림하여 정수로
        return res  # 반내림된 값


    def message(self, token, channel, text):  # slack에 메세지를 보낸다
        response = requests.post("https://slack.com/api/chat.postMessage",
                                 headers={"Authorization": "Bearer " + token},
                                 data={"channel": channel, "text": text}
                                 )

    def time_(self):  # 현시각 출력
        fseconds = time.time()
        second = int(fseconds % 60)
        fseconds //= 60
        minute = fseconds % 60
        fseconds //= 60
        hour = fseconds % 24
        hour = (hour + 9) % 24

        return hour, minute, second

    def is_repaint(self,ind_data,is_First): #리페인트 확인 함수 (다음 새로운 분봉일때 실행)

        if is_First==False: #실행이후 처음이 아닌경우
            len_ind = len(ind_data)
            if len_ind>10:
                for step in range(len_ind-10,len_ind):
                    if self.past_ind_data[step] != ind_data[step-1]:
                        print('지표 리페인트 발생')
                        print('이전 지표 길이:', len(self.past_ind_data), '\n현재 지표 길이:', len(ind_data))
                        print('이전 지표:',self.past_ind_data[step], '\n현재지표:', ind_data[step-1], '\n step : ',step)
                        print('이전 지표 전체:', self.past_ind_data, '\n현재 지표 전체:', ind_data)
                        break

                self.past_ind_data = ind_data #지표데이터 업데이트

            else:
                print('리페인트 확인 함수 에러: 비교하고자 하는 지표의 길이 설정값이 적음')

        else: # 실행이후 첫스탭인경우
            self.past_ind_data = ind_data




    def act(self, action, unit, ori_unit, real_data_count, short_or_long, print_message):  # 매매
        # 매매 포지션을 유지하고 오더된 유닛을 초과하지 않도록 설정
        hour, minute, second = self.time_()
        self.action = action
        self.unit = unit
        self.stock = ori_unit  # Env의 discrete_step 으로 변하기전 가진 주식수

        action_name = 'wait'


        if short_or_long == 'short':  # 숏이면 바뀌어야함(0일때 숏매수 2일때 롱매수 이므로)
            if action == 0:
                self.action = 2
                self.unit = [unit[2], 0, unit[0]]
                action_name = 'Sell short'

            if action == 2:
                self.action = 0
                self.unit = [unit[2], 0, unit[0]]
                action_name = 'Buy short'

        if short_or_long == 'long':  # 숏이면 바뀌어야함(0일때 숏매수 2일때 롱매수 이므로)
            if action == 0:
                action_name = 'Sell long'

            if action == 2:
                action_name = 'Buy long'

        if self.action == 0:  # 매도
            self.action_data[short_or_long].append(0)


        elif self.action == 1:  # 관망
            self.action_data[short_or_long].append(1)

        else:  # 매수
            self.action_data[short_or_long].append(2)




        if real_data_count[1] != self.past_data_date:  # 분봉 시간이 바꼈을때
            print('현시각:', hour, '시', minute, '분', second, '초')

            if action == 0:
                print('진입 분봉 시간:', real_data_count[1], '      행동: 매도', '       진입 갯수:', unit[0], '진입 가격:', self.agent.price)

            if action == 2:
                print('진입 분봉 시간:', real_data_count[1], '      행동: 매수', '       진입 갯수:', unit[2], '진입 가격:', self.agent.price)


            if short_or_long == 'long':
                print(
                    '------------------------------------------------------------------------------------------------------------------------------------------')
                print('APPO_LS long Agent   종목명:', self.symbol_name, '_', short_or_long, '           현재액션:', action, '_', action_name,
                      '       종목매수 가치(달러) : ', float(self.agent.price) * self.agent.stock, '    현재가(달러):',
                      float(self.agent.price), '   평단가:', 'None','       보유현금 : ',self.agent.cash, '    보유수량:', self.agent.stock)
                print(
                    '------------------------------------------------------------------------------------------------------------------------------------------')

    def recent_period(self, is_First):  # 최신데이터를 적절한 구간만큼 minute 고려하여 불러온다
        real_trading_data_num = 1000  # 새로출력시 불러올 데이터 갯수(새로운 분인경우)

        ##############현재 최신 데이터 시간 불러오기
        time.sleep(1)  # 많은호출 방지

        for i in range(20):  # 최대 20번 재시도
            try:
                cur_minute = 1
                period= 30
                ohlcv = get_price.fetch_and_save_data(params.trade_market,params.API_stock_name,cur_minute,period,self.access_token) #1분봉 360개(period=3은 120개*3)
                break  # 데이터를 성공적으로 가져오면 for 루프를 빠져나옴

            except ccxt.NetworkError:  # 네트워크 오류가 발생하면
                if i < 19:  # 19번 이하로 시도했다면
                    print(i + 1, '번 재시도')
                    time.sleep(60) # 60초마다 시도가능(token 호출 제한)
                    continue  # 다시

                else:  # 10번이 모두 실패했다면
                    raise  # 오류를 던짐

        latest_ohlcv = ohlcv['open'].iloc[-1]  # 가장 최근의 봉 데이터를 선택
        open_time = ohlcv['datetime'].iloc[-1]

        if is_First == False:
            try:  # 처음 불러온 저장데이터 호출
                csv_data = pd.read_csv('save_real_date_' + self.symbol_name.replace('/', '_')).values
                ori_last_minute = datetime.strptime(csv_data[-1][0][:16], "%Y-%m-%d %H:%M") # 저장된 데이터의 마지막 시각
            except:
                pass

            ########### minute 만큼 시간이 지났는지 확인
            time1 = open_time
            time2 = ori_last_minute
            time_diff = time1 - time2
            time_minute = int(time_diff.total_seconds() / 60)

            if round(time_minute / params.minute) < real_trading_data_num:  # 뽑게될 데이터 갯수가 적으면 데이터를 더뽑음
                last_minute = ori_last_minute - timedelta(minutes=params.minute * real_trading_data_num)  # 데이터 1000개만큼 출력되도록 시작점 설정
                last_minute = str(last_minute)[:16]
                time2 = datetime.strptime(last_minute, '%Y-%m-%d %H:%M')
                time_diff = time1 - time2
                time_minute = int(time_diff.total_seconds() / 60)

            ########## 백테스트 마지막 ~ 최신 데이터 호출 구간
            if time_minute % params.minute == 0:  # minute 만큼 시간이 지났을경우 새로운 데이터 호출
                is_API = 'API'
                real_data_count = [last_minute, open_time]
            else:
                is_API = 'csv'
                real_data_count = [last_minute, ori_last_minute]


        if is_First == True:  # csv도 없는 첫상태, 초기 트레이딩 시작일때
            is_API = 'API'
            csv_data = pd.read_excel('trading_times.xlsx').values
            ori_last_minute = csv_data[0][0][:16]  # 저장된 데이터의 처음시각
            real_data_count = [ori_last_minute, open_time]  # 종목의 최근까지 호출

        return real_data_count, is_API





    def start_trading(self):
        # 학습했던 구간의 데이터를 불러온다
        env = env_.Env()
        is_First=True #처음인경우
        real_data_count, is_API_ = self.recent_period(is_First)  #처음 뽑으면 API 최신까지 호출
        print(real_data_count,'원시 데이터 호출 날짜 기간')

        now= datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data_count= [params.data_count[0],now]
        data_ = env.stock_data_create(params.minute, data_count, self.access_token) # 이전 + 실시간 데이터 호출

        long_input_, short_input_, ori_ind_data= env.input_create(params.minute, params.ratio,real_data_count,
                                                               params.coin_or_stock, params.point_value,
                                                               params.short_ind_name, params.long_ind_name,
                                                               data_)  # ind에서 높은 값을 뽑음
        ind_data = [long_input_,short_input_,ori_ind_data]
        real_save_price_data(data_,ind_data, self.symbol_name)


        #############################실시간 데이터 호출
        self.if_First = True
        idx =0
        while True:  # 실시간 호출 및 액션
            # 재 접속 시도

            for step in range(100):
                try:
                    cur_minute = 1
                    period = 30
                    self.ohlcv = get_price.fetch_and_save_data(params.trade_market, params.API_stock_name, cur_minute,
                                                          period, self.access_token)  # 1분봉 360개(period=3은 120개*3)
                    break

                except Exception as e:
                    print(e)
                    print('재접속 시도', step, '번째')



            if real_data_count[1] != self.past_data_date or self.if_First == True:  # 분봉이 변한 경우 or 첫스탭인경우 실행
                res_data_root = 'traj/backtest_result.json'
                
                # JSON 파일을 읽음
                with open(res_data_root, 'r') as file:
                    json_data = file.read()

                # 문자열을 JSON 객체로 변환
                json_objects = json_data.strip().split('\n')
                json_array = [json.loads(obj) for obj in json_objects]

                # DataFrame으로 변환
                res_data = pd.DataFrame(json_array)

                is_API, data_total, ind_data = real_load_price_data(self.symbol_name, real_data_count)  # csv를 불러올지, api를 불러올지 선택
                long_input_ = ind_data[0]
                short_input_ = ind_data[1]
                ori_ind_data = ind_data[2]

                # 데이터 불러옴 (처음부터 갱신)
                if is_API == 'API':
                    data_ = env.stock_data_create(params.minute, data_count, self.access_token)  # 이전 + 실시간 데이터 호출

                    long_input_, short_input_, ori_ind_data = env.input_create(params.minute, params.ratio,
                                                                               real_data_count,
                                                                               params.coin_or_stock, params.point_value,
                                                                               params.short_ind_name,
                                                                               params.long_ind_name,
                                                                               data_)  # ind에서 높은 값을 뽑음


                ind_data = [long_input_, short_input_, ori_ind_data]
                real_save_price_data(data_, ind_data, self.symbol_name)

                long_train_data, long_val_data, long_test_data, long_ori_close, long_total_input, long_date_data, long_total_date = long_input_
                short_train_data, short_val_data, short_test_data, short_ori_close, short_total_input, short_date_data, short_total_date = short_input_

                self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])  # 실시간 가격., 지표 데이터들
                self.long_scale_input = long_total_input
                self.long_date_data = long_total_date

                self.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
                self.short_scale_input = short_total_input
                self.short_date_data = short_total_date

                long_res_data =self.long_scale_input #결과 최신 지표 데이터 저장
                short_res_data =self.short_scale_input


                ##########################에이전트 호출
                for short_or_long in params.short_or_long_data:  # 롱숏 에이전트 호출
                    # global net
                    Global_actor = NET.Global_actor
                    Global_critic = NET.Global_critic


                    window= params.window[short_or_long]

                    # 숏 or 롱 포지션 따라 인풋 정의
                    if short_or_long == 'short':
                        input_ = short_res_data
                        self.input_dim = params.input_dim['short']
                        ori_close = self.short_price_data
                        date_data = self.short_date_data

                    else:
                        input_ = long_res_data
                        self.input_dim = params.input_dim['long']
                        ori_close = self.long_price_data
                        date_data = self.long_date_data

                    self.Global_policy_net[short_or_long] = Global_actor('cpu', window, self.input_dim, short_or_long,params.Neural_net, params.bidirectional_)
                    self.Global_value_net[short_or_long] = Global_critic('cpu', window, self.input_dim, short_or_long,params.Neural_net, params.bidirectional_)


                    agent_num = 0  # 글로벌 에이전트 넘버=0
                    self.agent_data[short_or_long] = PPO_Agent.PPO(window,  # LSTM 윈도우 사이즈
                                                                   params.cash,  # 초기 보유현금
                                                                   params.cost,  # 수수료 %
                                                                   params.device,  # 디바이스 cpu or gpu
                                                                   params.k_epoch,  # K번 반복
                                                                   input_,  # 인풋 데이터
                                                                   ori_close,  # 주가 데이터
                                                                   date_data,  # 날짜 데이터
                                                                   self.input_dim,  # feature 수
                                                                   agent_num,
                                                                   params.coin_or_stock,
                                                                   params.deposit,
                                                                   params.slippage,
                                                                   short_or_long,  # 숏인지 롱인지
                                                                   self.Global_policy_net[short_or_long],  # 글로벌넷
                                                                   self.Global_value_net[short_or_long],  # 글로벌넷
                                                                   self.is_back_testing
                                                                   )

                    self.agent = self.agent_data[short_or_long]  # 에이전트 정의
                    self.policy_net = self.Global_policy_net[short_or_long] #폴리시 정의



                    # 상태 실시간 업데이트
                    if short_or_long =='short':
                        self.decide_action = self.agent.short_decide_action
                        self.discrete_step = env.short_discrete_step
                        self.agent.short_ori_input = ori_ind_data[1]
                        self.agent.price_data = self.short_price_data  # 가격

                        pos = self.balance2['info']['positions']  #숏 매수 물량
                        for position in pos:
                            if position['symbol'] == self.symbol_name.replace('/', ''):
                                short_unit = float(position['positionAmt']) #숏매수는 음수로 표시됨
                                if short_unit < 0 :
                                    self.agent.short_unit = np.abs(short_unit) #숏매수는 음수로 표시됨
                                else:
                                    self.agent.short_unit = 0 #롱 물량이 있으면 그냥 0 취급

                        self.ori_unit = self.agent.short_unit #discrete 로 변하기전 unit

                    if short_or_long =='long':
                        self.decide_action = self.agent.long_decide_action
                        self.discrete_step = env.long_discrete_step
                        self.agent.long_ori_input = ori_ind_data[0]
                        self.agent.price_data = self.long_price_data  # 가격 데이터

                    ##저장된 가중치 load
                    self.policy_net.load()

                    #policy 계산
                    with torch.no_grad():
                        prob = self.policy_net(self.agent.LSTM_input).to(self.agent.device)
                        policy = F.softmax(prob, dim=1)  # policy

                    self.agent.price = self.agent.price_data[-1]  # 현재(최근) 주가업데이트
                    self.agent.back_testing = True

                    #액션 설정
                    action, unit= self.decide_action(policy[-1],deposit=1)
                    step = len(self.agent.PV_list)-1
                    action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트

                    print(res_data, '@@@@@@@@@@@@@@@@@@@')
                    print(res_data.loc[res_data['index'] == 'action_data'].iloc[0], 'fasknfkansfasfnklaklsf')

                    if action == 0:  # 매도
                        #데이터 저장
                        res_data.loc[res_data['index'] == 'action_data'].values[0][0][short_or_long].append(0)
                        res_data.loc[res_data['index'] == 'sell_data'][short_or_long].append(self.agent.price_data[step])
                        res_data.loc[res_data['index'] == 'sell_date'][short_or_long].append(step)

                    elif action == 1:  # 관망
                        res_data.loc[res_data['index'] == 'action_data'][short_or_long].append(1)

                    else:  # 매수
                        # 데이터 저장
                        res_data.loc[res_data['index'] == 'action_data'][short_or_long].append(2)
                        res_data.loc[res_data['index'] == 'buy_data'][short_or_long].append(self.agent.price_data[step])
                        res_data.loc[res_data['index'] == 'buy_date'][short_or_long].append(step)

                    # 데이터 저장
                    res_data.loc[res_data['index'] == 'PV_data'][short_or_long].append(self.agent.PV-self.agent.init_cash)
                    res_data.loc[res_data['index'] == 'price_data'][short_or_long].append(self.agent.price_data[step])

                    # JSON 파일로 저장
                    res_data.to_json('traj/backtest_result.json', orient='records', lines=True)


                    print(step + 1, '/', len(self.agent.price_data), '전진분석 테스팅중..', short_or_long + '_agent PV :',
                          float(self.PV_data[short_or_long][-1]))
                    print(policy[step], '스탭 폴리시', action, '액션', unit, '유닛', self.agent.stock, '보유주식수')

                    self.print_message = True # act에서 메세지 프린트
                    self.is_repaint(short_res_data[0][-40:], self.if_First)  # 지표의 리페인트 확인

                    #self.act(action, unit, self.ori_unit ,real_data_count, short_or_long, self.print_message) #real_data_count = 최근 기간 문자열

            self.past_data_date = real_data_count[1] #최신 분봉 데이터 업데이트
            self.print_message =False #프린트
            self.if_First = False #처음인지
            idx+=1
            time.sleep(0.1)




if __name__ == '__main__':
    stock_env = stock_Env()
    stock_env.start_trading()




