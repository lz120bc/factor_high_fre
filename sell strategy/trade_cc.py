import os
import pandas as pd

trade_path = 'D:\\中信建投实习\\bigdata\\trade'
files_name = []
trade_data = []
for root, dirs, files in os.walk(trade_path):
    for fi in files:
        path = os.path.join(root, fi)
        files_name.append(path)
for i in files_name:
    if '20230601_20230609' in i:
        trd = pd.read_feather(i)
        trd = trd[trd['trade_bs_flag'] != 'C'] #C为撤单
        trd = trd[['securityid', 'date', 'time', 'trade_price', 'trade_volume']]
        trd['time'] = pd.to_datetime(trd['time'] / 1000, format='%H%M%S')
        trd = trd.groupby(['securityid', 'date', pd.Grouper(key='time', freq='3S'), 'trade_price'
                           ])['trade_volume'].sum().reset_index() #trade数据是毫秒级，需手动分成3秒tick
        trd = trd.sort_values(['date', 'time', 'trade_price'])
        trade_data.append(trd)
trade_data = pd.concat(trade_data).reset_index()
trade_data.to_feather(trade_path + '\\trade.feather')

# 高频数据处理中，处理数据频率是难题
#Oder数据（包含每一笔交易，index为用户ID）可进行对手分析，交易对手风格描写，分析庄家风格和行为，可以直接与其交易或做对手方；也可根据特定风格寻找对手；
# 也可按个单的多少和金额分析散户/庄家的多少。如果散户较多，则市场化明显，可通过量化进行有效价格预测