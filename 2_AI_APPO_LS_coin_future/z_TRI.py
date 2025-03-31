import numpy
impot xe_train as params
import a_ind as ind_

db.execute(
    f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol={self.name} ORDER BY datetime DESC limit {data_count}) as foo order by datetime asc;")
data_set = db.fetchall()
data_set = self.time_Frame(data_set, minute)

ind=ind_.ind_env()

class TRI_env:

    def __init__(self,data,cci기간값1,vMultipleH_TRI,LRMultiple1):
        self.data1=data
        self.data2='달러인덱스'

        self.cci기간값1= cci기간값1
        self.vMT=vMultipleH_TRI
        self.LRM=LRMultiple1


    def NQSNP_TRI(self):
        vData1_CCI180,ori_vData1_CCI180=ind.CCI(self.data1,self.cci기간값1)
        vUSD_CCI180,ori_vData1_CCI180=ind.CCI(self.data2,self.cci기간값1)

        ma1,ori_ma1=ind.MA(vData1_CCI180,self.cci기간값1)
        ma2,ori_ma2=ind.MA(vUSD_CCI180,self.cci기간값1)

        std1,ori_std1= ind.STD(vData1_CCI180,self.cci기간값1)
        std2,ori_std2= ind.STD(vUSD_CCI180,self.cci기간값1)


        Data1_std=(ori_vData1_CCI180-ori_ma1)/ori_std1
        USD_std=(ori_vUSD_CCI180-ori_ma2)/ori_std2

        vFinalAvg=Data1_std-USD_std*3
        vTRI=vFinalAvg.rolling(self.vMT).sum()

        LR_TRI,ori_LR_TRI=ind.slope_line(vTRI,self.LRM)

        return LR_TRI, ori_LR_TRI


