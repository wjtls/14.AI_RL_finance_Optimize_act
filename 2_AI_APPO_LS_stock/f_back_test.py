import pandas as pd

import e_train as params
import a_Env as env_
import c_PPO_Agent as PPO_Agent
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import b_network as NET
import torch.multiprocessing as multiprocessing
from datetime import datetime

import random
import torch
seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

env=env_.Env()




class back_testing:
    def __init__(self,train_val_test,path):
        # 시뮬 data
        self.is_back_testing=True
        self.window=0

        # traj위치 경로
        self.path = path

        # long short 2개 생성
        self.PV_data = {'long':[], 'short':[]}
        self.PV_return_data = {'long':[], 'short':[]} #PV의 누적 수익률
        self.PV_log_return_data = {'long':[], 'short':[]} # PV의 로그수익률 저장
        self.PV_Cumul_return_data = {'long':[], 'short':[]}  #PV 누적 로그수익률
        self.pv_log_cumul_return = {'long':[], 'short':[]} #PV 누적 로그수익률을 수익률로 변환
        self.Agent_policy = {'long':[], 'short':[]} # 에이전트의 가장높은 폴리시 확률 1개 저장
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buysell_ratio = {'long':[], 'short':[]} # 매매 비중(소유대비)
        self.buy_date = {'long':[], 'short':[]} #매수 날짜
        self.sell_date = {'long':[], 'short':[]} #매도 날짜
        self.price_data={'long':[],'short':[]} #가격 데이터
        self.date_data={'long':[], 'short':[]} # 날짜 데이터
        self.scale_input={'long':[],'short':[]}
        self.agent_stock = {'long':[], 'short':[]} # 보유 수량

        self.train_val_test = train_val_test

        self.Global_policy_net={}
        self.Global_value_net={}
        self.agent_data={}


        # 데이터 호출
        env = env_.Env()

        is_API, data_ , ind_data = env_.load_price_data(params.API_stock_name)  # csv를 불러올지, api를 불러올지 선택

        long_input_ = ind_data[0]
        short_input_= ind_data[1]
        ori_ind_data = ind_data[2]

        # 현재 시간 가져오기
        now = datetime.now()

        # 트레이딩시 초기 날짜
        days_ago = pd.read_excel(f'{self.path}/trading_times.xlsx')
        days_ago = days_ago.values[0][0]

        # 백테스트 날짜 지정
        data_count = [
            days_ago,
            now.strftime('%Y-%m-%d %H:%M')
        ]
        '''
        data_count = [
            '2024-09-01 00:00',
            now.strftime('%Y-%m-%d %H:%M')
        ]
        '''



        if is_API == 'API':
            if params.coin_or_stock == 'stock':
                from a_korea_invest_api_env import get_ovsstk_chart_price  # 해외 현물 가격( 최근 1달)
                ACCESS_TOKEN, ACCESS_TOKEN_EXPIRED = get_ovsstk_chart_price.get_access_token()  # 한투 API접속

                data_ = env.stock_data_create(params.minute, data_count, ACCESS_TOKEN)
            else:
                data_ = env.coin_data_create(params.minute, data_count, params.real_or_train, params.coin_or_stock,
                                         params.point_value, params.API_stock_name)  # 학습시 뽑은 history 데이터

            long_input_, short_input_, ori_ind_data = env.input_create(params.minute, params.ratio, data_count,
                                                                       params.coin_or_stock, params.point_value,
                                                                       params.short_ind_name, params.long_ind_name,
                                                                       data_)  # ind에서 높은 값을 뽑음

            ind_data = [long_input_, short_input_, ori_ind_data]
            env_.save_price_data(data_, ind_data, params.API_stock_name)


        long_train_data,long_val_data,long_test_data,long_ori_close,long_total_input,long_date_data,long_total_date= long_input_
        short_train_data,short_val_data,short_test_data,short_ori_close,short_total_input,short_date_data,short_total_date= short_input_


        # APPO는 전체 넣음, PPO는 데이터셋 나눠서 넣음

        if train_val_test == 'train':
            self.long_price_data = long_ori_close[0]
            self.long_scale_input = long_train_data
            self.long_date_data = long_date_data[0]

            self.short_price_data = short_ori_close[0]
            self.short_scale_input = short_train_data
            self.short_date_data = short_date_data[0]


        elif train_val_test == 'val':
            self.long_price_data = long_ori_close[1]
            self.long_scale_input = long_val_data
            self.long_date_data = long_date_data[1]

            self.short_price_data = short_ori_close[1]
            self.short_scale_input = short_val_data
            self.short_date_data = short_date_data[1]


        elif train_val_test == 'test':
            self.long_price_data = long_ori_close[2]
            self.long_scale_input = long_test_data
            self.long_date_data = long_date_data[2]

            self.short_price_data = short_ori_close[2]
            self.short_scale_input = short_test_data
            self.short_date_data = short_date_data[2]


        elif train_val_test == 'total':
            self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
            self.long_scale_input = long_total_input
            self.long_date_data = long_total_date

            self.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
            self.short_scale_input = short_total_input
            self.short_date_data = short_total_date



        # 에이전트 호출

        for short_or_long in params.short_or_long_data:
            if short_or_long=='long':
                input_dim=len(params.long_ind_name)
                self.window = params.window[short_or_long]
            if short_or_long=='short':
                input_dim=len(params.short_ind_name)
                self.window = params.window[short_or_long]

            # global net
            Global_actor = NET.Global_actor
            Global_critic = NET.Global_critic
            self.Global_policy_net[short_or_long]=Global_actor(params.device, self.window,input_dim, short_or_long, params.Neural_net, params.bidirectional_)  #dict 에 글로벌넷 저장
            self.Global_value_net[short_or_long]=Global_critic(params.device, self.window,input_dim, short_or_long, params.Neural_net, params.bidirectional_)  #Global_policy, Global_value ,


            #숏 or 롱 포지션 따라 인풋 정의
            if short_or_long=='short':
                input_=self.short_scale_input
                self.input_dim=params.input_dim['short']
                ori_close=self.short_price_data
                date_data=self.short_date_data

            else:
                input_=self.long_scale_input
                self.input_dim = params.input_dim['long']
                ori_close=self.long_price_data
                date_data=self.long_date_data

            if params.train_stock_or_future == 'future':
                cost_ = params.cost  # 선물 cost
            else:
                cost_ = params.stock_cost  # 주식 cost

            agent_num = 0  # 글로벌 에이전트 넘버=0
            #에이전트 정의
            self.agent_data[short_or_long] = PPO_Agent.PPO(self.window,  # LSTM 윈도우 사이즈
                                                      params.cash,  # 초기 보유현금
                                                      cost_,  # 수수료 %
                                                      params.device,  # 디바이스 cpu or gpu
                                                      params.k_epoch,  # K번 반복
                                                      input_,  # 인풋 데이터
                                                      ori_close,  # 주가 데이터
                                                      date_data,  # 날짜 데이터
                                                      self.input_dim,  # feature 수
                                                      agent_num,
                                                      params.coin_or_stock,
                                                      params.deposit,
                                                      params.backtest_slippage,
                                                      short_or_long,  # 숏인지 롱인지
                                                      self.Global_policy_net[short_or_long],  # 글로벌넷
                                                      self.Global_value_net[short_or_long],  # 글로벌넷
                                                      self.is_back_testing
                                                      )



    def reset(self):
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date={'long':[], 'short':[]}
        self.sell_date= {'long':[], 'short':[]}



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

        if params.train_stock_or_future != 'future':
            self.decide_action = self.agent.SC_decide_action
            self.discrete_step = self.agent.SC_discrete_step
            print('주식처럼 백테스트')

        ##저장된 가중치 load
        policy_net.load()

        # back testing

        for step in range(len(self.agent.price_data)):
            with torch.no_grad():
                prob_ = policy_net(self.agent.LSTM_input.to(self.agent.device)).to(self.agent.device)
                policy = F.softmax(prob_,dim=1)  # policy
            self.agent.price = self.agent.price_data[step]  # 현재 주가업데이트

            if params.train_stock_or_future == 'future':  # 선물처럼 학습시킬경우
                action, unit = self.decide_action(policy[step],params.deposit)  # 액션 선택
            else:
                action, unit = self.decide_action(policy[step])

            self.past_stock = self.agent.stock #이전 수량 저장
            self.past_cash = self.agent.cash #이전 현금 저장
            action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트


            if step == 10:
                print(policy[:10], 'agent1 10개')
                Agent_data_num= round(len(policy)/params.Agent_num)
                print(policy[Agent_data_num:Agent_data_num + 10], 'agent2 10개',Agent_data_num,'스탭 부터')
                print(policy[Agent_data_num * 2:Agent_data_num * 2 + 10], 'agent3 10개',Agent_data_num* 2,'스탭 부터')
                print(policy[Agent_data_num * 3:Agent_data_num * 3 + 10], 'agent4 10개')
                print(policy[Agent_data_num * 4:Agent_data_num * 4 + 10], 'agent5 10개')
                print(policy[Agent_data_num * 5:Agent_data_num * 5 + 10], 'agent6 10개')
                print(policy[Agent_data_num * 9:Agent_data_num * 9 + 10], 'agent10 10개')
                print(policy[Agent_data_num * 10:Agent_data_num * 10 + 10], 'agent11 10개')
                print(policy[Agent_data_num * 12:Agent_data_num * 12 + 10], 'agent13 10개')
                print(policy[Agent_data_num * 15:Agent_data_num * 15 + 10], 'agent16 10개')
                print(policy[Agent_data_num * 16:Agent_data_num * 16 + 10], 'agent17 10개')
                print(policy[Agent_data_num * 17:Agent_data_num * 17 + 10], 'agent18 10개')
                print(policy[Agent_data_num * 73:Agent_data_num * 73 + 10], 'agent74 10개')

                max_values = policy.max(dim=0).values  # 각 열에서 가장 큰 값들을 찾습니다.
                max_policy = max_values.tolist()  # policy 값으로 변환하여 출력합니다.
                min_values = policy.min(dim=0).values  # 각 열에서 가장 큰 값들을 찾습니다.
                min_policy = min_values.tolist()  # policy 값으로 변환하여 출력합니다.
                print(max_policy,'가장큰값')
                print(min_policy, '가장작은값')


            if action == 0: #매도
                self.buysell_ratio[is_short_or_long].append(((self.past_stock-self.agent.stock)/self.past_stock)*100 if self.past_stock >0 else 0) #매도 비중 : (이전보유주식수 - 현재주식수)/이전보유주식수
                self.action_data[is_short_or_long].append(0)
                if unit[0] !=0 : #매매 유닛이 0이 아닌경우
                    self.sell_data[is_short_or_long].append(self.agent.price_data[step])
                    self.sell_date[is_short_or_long].append(self.agent.date_data[step])

            elif action == 1: #관망
                self.buysell_ratio[is_short_or_long].append(0) #매매 비중
                self.action_data[is_short_or_long].append(1)

            else: #매수
                self.buysell_ratio[is_short_or_long].append(((self.past_cash-self.agent.cash)/self.past_cash)*100 if self.past_cash >0 else 0) # 매수 비중 : 매수액션에서 (이전캐시 - 현재캐시)/이전캐시
                self.action_data[is_short_or_long].append(2)
                if unit[2] != 0:  # 매매 유닛이 0이 아닌경우
                    self.buy_data[is_short_or_long].append(self.agent.price_data[step])
                    self.buy_date[is_short_or_long].append(self.agent.date_data[step])

            # 데이터 저장
            self.PV_data[is_short_or_long].append(self.agent.PV-self.agent.init_cash) #PV
            self.agent_stock[is_short_or_long].append(self.agent.stock) #보유수량
            self.Agent_policy[is_short_or_long].append(max(policy[step])) # 에이전트의 가장높은 폴리시 확률 1개 저장 -> 액션 확률 표시하기위함

            #PV의 누적 수익률 저장
            cumulative_return = ((self.agent.PV / self.agent.init_cash) - 1)*100
            self.PV_return_data[is_short_or_long].append(cumulative_return)


            # PV의 로그수익률 저장
            current_PV = self.agent.PV - self.agent.init_cash

            if step != 0:
                log_return = np.log(current_PV / self.PV_data[is_short_or_long][step-1])
                if  self.PV_data[is_short_or_long][step-1] <= 0 : #로그수익률이 null인경우
                    self.PV_log_return_data[is_short_or_long].append(0)
                else:
                    self.PV_log_return_data[is_short_or_long].append(log_return)

                # 누적 로그 수익률 계산
                if self.PV_data[is_short_or_long][step-1] <= 0 :  # 현재 로그수익률이 null인경우
                    self.PV_Cumul_return_data[is_short_or_long].append(0)
                else:
                    cumulative_log_return = self.PV_Cumul_return_data[is_short_or_long][-1] + log_return  # 이전 누적 로그 수익률에 현재 로그 수익률 추가
                    self.PV_Cumul_return_data[is_short_or_long].append(cumulative_log_return)  # 누적 로그 수익률 저장

            else: #처음이면
                self.PV_Cumul_return_data[is_short_or_long].append(0)
                self.PV_log_return_data[is_short_or_long].append(0)


            if step % 50 == 0 or step ==(len(self.agent.price_data)-1):  #실시간 출력값
                print(step + 1, '/', len(self.agent.price_data), '테스팅중..',  is_short_or_long + '_agent PV :', float(self.PV_data[is_short_or_long][-1]) )
                print(policy[step],'스탭 퐆리시',action,'액션',unit,'유닛',self.agent.stock,'보유주식수')

        # PV의 누적 로그 수익률의 실제 수익률 (실제 수익률을 추정 -> 데이터를 정규화하는것에 가까움)
        PV_log_returns_res = [np.exp(float(r) if r is not None else 0) - 1 for r in self.PV_Cumul_return_data[is_short_or_long]] # None 값을 0으로 바꾸고, float로 변환한 후 실제 수익률 계산
        self.pv_log_cumul_return[is_short_or_long] = PV_log_returns_res



        market_first = self.agent.price_data[0]
        market_last = self.agent.price_data[-1]

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

        self.date_data[is_short_or_long]=self.agent.date_data
        self.price_data[is_short_or_long]= self.agent.price_data
        self.scale_input[is_short_or_long]= self.agent.scale_input
        #print(len(self.PV_data),len(self.action_data),len(self.buy_data),len(self.buy_date),len(self.sell_date),len(self.sell_date),'aksfnkasnfklsnkf')

        print((((market_last / market_first) - 1) * 100).item(), ':Market ratio of long return')
        print(float(((self.PV_data[is_short_or_long][-1] + self.agent.init_cash) / (self.PV_data[is_short_or_long][0] + self.agent.init_cash) - 1) * 100),'% :' + is_short_or_long + '_agent PV return')

        if params.coin_or_stock=='future': #선물인경우
            print(float((((self.PV_data[is_short_or_long][-1]-self.agent.init_cash) / self.agent.deposit)) * 100),'% :' + is_short_or_long + '_agent 증거금 대비 PV return')

        return long_res,short_res



    def mul_back_test(self):  # 병렬 백테스트( 숏 롱 )
        process_list = []
        res_data={}
        long_res=multiprocessing.Queue()  #결과 저장
        short_res=multiprocessing.Queue()

        print('병렬 백테스트 시작.', '코어수:', multiprocessing.cpu_count())


        if params.backtest_hedge_on_off=='on':
            if params.multi_or_thread=='multi': #멀티 프로세싱인경우 (메모리 많이잡아먹음)
                if __name__ == '__main__':
                    for is_short_or_long in params.short_or_long_data:  # func 목록 모두 저장
                        proc = multiprocessing.Process(target=self.back_test,args=([is_short_or_long, long_res,short_res]))
                        process_list.append(proc)
                    [process.start() for process in process_list]

                    for step in range(len(params.short_or_long_data)): #숏, 롱 데이터 큐에 저장
                        is_short_or_long = params.short_or_long_data[step]
                        if is_short_or_long == 'long':
                            res_ = long_res.get()[0]
                        if is_short_or_long == 'short':
                            res_ = short_res.get()[0]
                        res_data[is_short_or_long] = res_

                    [process.join() for process in process_list] #종료

            else: #멀티프로세싱 아닌경우
                long_res_ = []
                short_res_ = []
                for is_short_or_long in params.short_or_long_data: #스레드 시뮬레이션 ( 메모리 낮은경우 대비)
                    long_res,short_res=self.back_test(is_short_or_long,long_res_,short_res_)
                    try:
                        if is_short_or_long == 'long':
                            res_ = long_res[0]
                        if is_short_or_long == 'short':
                            res_ = short_res[0]
                        res_data[is_short_or_long] = res_
                    except:
                        res_data=0

        return res_data


#추가해야될 기능 ,슬리피지, 월간 일간 주간 ,손익계산, 승률 , 거래횟수 , MDD, 롱수익 , 숏 수익
#로컬이 학습하고 옮긴다음 검증?
#CCI Neo 가지고 학습하는데



from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as TK
def start_backtest(path):

    # 결과 출력
    bk = back_testing(params.back_train_val_test,path)
    res_data=bk.mul_back_test()

    # res_data 저장
    backtest_obj = res_data['long']
    data_dict = backtest_obj.__dict__


    def simplify_data(value):
        # 모든 키를 반환하도록 수정
        if isinstance(value, list):
            return value[0] if value else None
        elif isinstance(value, dict):
            return {k: v for k, v in value.items()}  # 모든 키를 선택
        else:
            return value


    # 필요한 데이터만 선택
    filtered_data_dict = {key: simplify_data(value) for key, value in data_dict.items()}
    res_df = pd.DataFrame.from_dict(filtered_data_dict, orient='index').reset_index()

    # JSON으로 변환할 수 있도록 데이터 변환
    def convert_to_serializable(value):
        if isinstance(value, list):
            return [convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {key: convert_to_serializable(val) for key, val in value.items()}
        elif hasattr(value, 'tolist'):  # 텐서 객체를 리스트로 변환
            return value.tolist()
        else:
            return str(value)  # 기본적으로 문자열로 변환


    # 데이터프레임의 각 항목을 변환
    for col in res_df.columns:
        res_df[col] = res_df[col].apply(convert_to_serializable)


    # JSON 파일로 저장
    res_df.to_json(f'{path}/traj/backtest_result.json', orient='records', lines=True)

if __name__=='__main__':
    path=''
    start_backtest(path)

