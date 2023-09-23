from funs2 import *

# 市值处理
dsm = pd.read_csv('/Users/lvfreud/Desktop/中信建投/因子/data/TRD_Dalyr.csv')  # 市值文件
dsm.drop_duplicates(subset=['Stkcd'], inplace=True, keep='first')
dsm['dsmv'] = np.log(dsm['Dsmvtll'] / 100000.0)

# tick数据预处理
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick2'
fac = ['voi_neutral', 'sori_neutral', 'pearson_neutral', 'mpc_skew_neutral', 'bam_neutral', 'por_neutral']
files_name = []
data_sum = []
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
    if "20230504_20230707" in i:
        data = pd.read_feather(i)
        data['Stkcd'] = data.securityid.astype('int')
        data = data.merge(dsm, on='Stkcd', how='left')
        group_index = ['securityid', 'date', 'time']
        data.drop(data[data['eq_trading_phase_code'] != 'T'].index, inplace=True)
        data.drop(data.columns[~data.columns.isin(col)], axis=1, inplace=True)
        data.loc[data['offer_price1'] == 0, 'offer_price1'] = np.nan
        data.loc[data['bid_price1'] == 0, 'bid_price1'] = np.nan
        data['price'] = (data['offer_price1'] + data['bid_price1']) / 2
        data.loc[data['price'].isna(), 'price'] = data['last']
        data['minutes'] = (data['time'] / 100000).astype('int')
        data.drop(data[data['minutes'] < 930].index, inplace=True)
        data.drop(data[data['minutes'] > 1457].index, inplace=True)
        data['const'] = 1
        data.sort_values(group_index, inplace=True)
        vol = []
        val = []
        for date, g in data.groupby('date'):
            volumes = g['total_volume_trade'] - g['total_volume_trade'].shift(1)
            values = g['total_value_trade'] - g['total_value_trade'].shift(1)
            volumes.iloc[0] = g.iloc[0]['total_volume_trade']
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
        sec = tick.dt.second
        sec = sec % 3
        tick.loc[sec == 1] = tick[sec == 1] + datetime.timedelta(seconds=2)
        tick.loc[sec == 2] = tick[sec == 2] + datetime.timedelta(seconds=1)
        tick = tick.dt.time
        data = pd.concat([data, tick], axis=1)
        data.drop_duplicates(subset=['securityid', 'date', 'tick'], inplace=True, keep='last')
        data.reset_index(drop=True, inplace=True)
        data_sum.append(data)

data = pd.concat(data_sum).reset_index(drop=True)
del data_sum
ws = 5*20
data = tick_handle(data, ws)
data.sort_values(['securityid', 'date', 'time'], inplace=True)

# 输出标准化因子值
dar = []
for (date, time), group in data.groupby(['date', 'tick']):
    g = pd.DataFrame(index=group.index)
    for factor in fac:
        g[factor + '_rank'] = group[factor].rank(ascending=False) / len(g)
    dar.append(g)
dar = pd.concat(dar, axis=0)
data = pd.concat([data, dar], axis=1)
del dar

fac = ['voi_neutral_rank', 'sori_neutral_rank', 'pearson_neutral_rank',
       'mpc_skew_neutral_rank', 'bam_neutral_rank', 'por_neutral_rank']
ric = [5.1, 9.2, 1.0, 1.9, 2.5, 2.6]
data['port'] = (data[fac] * ric).dropna().sum(axis=1) / np.sum(ric)
data.reset_index().to_feather(working_path+'/tickf.feather')

# chg = []
# for (date, sec), group in data.groupby(['date', 'securityid']):
#     if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any():
#         continue
#     if date < 20230601:
#         price1 = twap(group)
#         price2, remain, ratio = easy_test(group, factor='port')
#         chg.append([sec, date, ratio])
# chg = pd.DataFrame(chg, columns=['sec', 'date', 'ratio'])
# chg = chg.groupby('sec')['ratio'].mean().reset_index()
# chg.to_csv('model1.csv', index=False)
