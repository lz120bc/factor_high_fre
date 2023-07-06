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
        trd = trd[trd['trade_bs_flag'] != 'C']
        trd = trd[['securityid', 'date', 'time', 'trade_price', 'trade_volume']]
        trd['time'] = pd.to_datetime(trd['time'] / 1000, format='%H%M%S')
        trd = trd.groupby(['securityid', 'date', pd.Grouper(key='time', freq='3S'), 'trade_price'
                           ])['trade_volume'].sum().reset_index()
        trd = trd.sort_values(['date', 'time', 'trade_price'])
        trade_data.append(trd)
trade_data = pd.concat(trade_data).reset_index()
trade_data.to_feather(trade_path + '\\trade.feather')
