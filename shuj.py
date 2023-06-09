import pandas as pd
import os
import numpy as np
import datetime

# data = pd.read_csv('E:\\data\\tick.csv', low_memory=False)
# working_path = 'E:\\data\\tick'
# dsm = pd.read_csv('E:\\data\\TRD_Dalyr.csv')
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
dsm = pd.read_csv('/Users/lvfreud/Desktop/中信建投/因子/data/TRD_Dalyr.csv')
dsm.drop_duplicates(subset=['Stkcd'], inplace=True, keep='first')
dsm['dsmv'] = np.log(dsm['Dsmvtll'] / 100000.0)
files_name = []
data = []
bao = []
for i in range(1, 6):
    bao.append('offer_price' + str(i))
    bao.append('offer_volume' + str(i))
    bao.append('bid_price' + str(i))
    bao.append('bid_volume' + str(i))
col = ['securityid', 'date', 'time', 'high', 'low', 'last', 'total_value_trade',
       'total_volume_trade', 'num_trades', 'dsmv'] + bao
for root, dirs, files in os.walk(working_path):
    for fi in files:
        path = os.path.join(root, fi)
        files_name.append(path)
for i in files_name:
    if "20230424_20230504" in i:
        data.append(pd.read_feather(i))
data = pd.concat(data).reset_index(drop=True)
data['securityid'] = data.securityid.astype('int')
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
data = data.merge(dsm, left_on='securityid', right_on='Stkcd', how='left')
group_index = ['securityid', 'date', 'time']
data.drop(data[data['eq_trading_phase_code'] != 'T'].index, inplace=True)
data.drop(data.columns[~data.columns.isin(col)], axis=1, inplace=True)
data.loc[data['offer_price1'] == 0, 'offer_price1'] = np.nan
data.loc[data['bid_price1'] == 0, 'bid_price1'] = np.nan
data['price'] = (data['offer_price1'] + data['bid_price1']) / 2
data.loc[data['price'].isna(), 'price'] = data['last']
data['minutes'] = (data['time'] / 100000).astype('int')
data.drop(data[data['minutes'] < 930].index, inplace=True)
data['const'] = 1
data.sort_values(group_index, inplace=True)
vol = []
val = []
for (sec, date), g in data.groupby(['securityid', 'date']):
    volumes = g['total_volume_trade'] - g['total_volume_trade'].shift(1)
    values = g['total_value_trade'] - g['total_value_trade'].shift(1)
    vol.append(volumes)
    val.append(values)
vol = pd.concat(vol, axis=0)
val = pd.concat(val, axis=0)
vol.name = 'volumes'
val.name = 'values'
data = pd.concat([data, vol, val], axis=1)
tick = pd.to_datetime(data['time'] / 1000, format="%H%M%S")
tick.name = 'tick'
sec = tick.dt.second
sec = sec % 3
tick.loc[sec == 1] = tick[sec == 1] + datetime.timedelta(seconds=2)
tick.loc[sec == 2] = tick[sec == 2] + datetime.timedelta(seconds=1)
tick = tick.dt.time
data = pd.concat([data, tick], axis=1)
data.drop_duplicates(subset=['securityid', 'date', 'tick'], inplace=True, keep='last')
data.reset_index(drop=True, inplace=True)

data.to_feather(working_path+'/tickda.feather')
