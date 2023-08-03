import os
import datetime
from funs2 import *
import threading

# 市值处理
dsm = pd.read_csv('D:\\中信建投实习\\中信实习-算法交易\\因子计算回测\\data\\TRD_Dalyr.csv',sep='\s+')  # 市值文件
dsm.drop_duplicates(subset=['Stkcd'], inplace=True, keep='first')
dsm['dsmv'] = np.log(dsm['Dsmvtll'] / 100000.0)

# tick数据预处理
working_path = 'D:\\中信建投实习\\bigdata\\tick'
fac = ['voi_neutral', 'sori_neutral', 'pearson_neutral', 'mpc_skew_neutral', 'bam_neutral', 'por_neutral']
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
    if "20230601_20230609" in i:
        data.append(pd.read_feather(i))
data = pd.concat(data).reset_index(drop=True)
data['Stkcd'] = data.securityid.astype('int')
data = data.merge(dsm, on='Stkcd', how='left')
group_index = ['securityid', 'date', 'time']
data.drop(data[data['eq_trading_phase_code'] != 'T'].index, inplace=True)
data.drop(data.columns[~data.columns.isin(col)], axis=1, inplace=True)
#tick价值表示
data.loc[data['offer_price1'] == 0, 'offer_price1'] = np.nan #涨停
data.loc[data['bid_price1'] == 0, 'bid_price1'] = np.nan #跌停
data['price'] = (data['offer_price1'] + data['bid_price1']) / 2 #中间价
data.loc[data['price'].isna(), 'price'] = data['last'] #如果涨停或跌停，用最新价代替
#时间
data['minutes'] = (data['time'] / 100000).astype('int')
data.drop(data[data['minutes'] < 930].index, inplace=True)
data.drop(data[data['minutes'] > 1457].index, inplace=True) # 14：57-15：00是收盘竞价，没有挡位，可以下单不可以撤单，一般14：57-15：00tick都不可见，在15：00还会产生最后一个tick
data['const'] = 1
data.sort_values(group_index, inplace=True)
vol = []
val = []
#算每天每个股票的每一tick成交价格和量（用下一个减去上一个）
for (sec, date), g in data.groupby(['securityid', 'date']):
    volumes = g['total_volume_trade'] - g['total_volume_trade'].shift(1)
    values = g['total_value_trade'] - g['total_value_trade'].shift(1)
    volumes.iloc[0] = g.iloc[0]['total_volume_trade'] #第一个不变
    values.iloc[0] = g.iloc[0]['total_value_trade']
    vol.append(volumes)
    val.append(values)
vol = pd.concat(vol, axis=0)
val = pd.concat(val, axis=0)
vol.name = 'volumes'
val.name = 'values'
data = pd.concat([data, vol, val], axis=1)
tick = pd.to_datetime(data['time'] / 1000, format="%H%M%S")
tick.name = 'tick'
# 上证所和深证所发来的tick（快照）数据不全部以三秒为区间，且时间并不一致，所以要都变成3秒间隔且时间一直的tick。例如：2秒的tick；9：30：01的tick
sec = tick.dt.second
sec = sec % 3
tick.loc[sec == 1] = tick[sec == 1] + datetime.timedelta(seconds=2)
tick.loc[sec == 2] = tick[sec == 2] + datetime.timedelta(seconds=1)
tick = tick.dt.time
data = pd.concat([data, tick], axis=1)
data.drop_duplicates(subset=['securityid', 'date', 'tick'], inplace=True, keep='last')
data.reset_index(drop=True, inplace=True)
# 可以改tick区间（频率）获得不同效果

lock = threading.Lock()
ws = 5*20 #五分钟tick量
data = tick_handle(data, ws)
data.sort_values(['securityid', 'date', 'time'], inplace=True)

# 输出标准化因子值
dar = []
for (date, time), group in data.groupby(['date', 'tick']): #横向比对（固定日期、时间，找各股信息），每个group是每个tick下全部股票信息
    g = pd.DataFrame(index=group.index)
    for factor in fac:
        g[factor + '_rank'] = group[factor].rank(ascending=False) / len(g) #排序除以总数，标准化
    dar.append(g)
dar = pd.concat(dar, axis=0) #把list里的dataframe变成一个dataframe
data = pd.concat([data, dar], axis=1) #把dar加进data
del dar
data.to_feather(working_path+'\\tickf.feather')
