import matplotlib.pyplot as plt
import os
from cul_funs import *
import threading
import time

# glob_f = ['voi', 'rwr', 'peaks', 'vc', 'skew', 'kurt', 'disaster', 'pearson', 'mpb', 'pob']
glob_f = ['rwr', 'pearson']
factor_ic = []
col = ['securityid', 'date', 'time', 'high', 'low', 'last', 'total_value_trade',
       'total_volume_trade', 'offer_price1', 'bid_price1', 'offer_volume1', 'bid_volume1']

lock = threading.Lock()

# data = pd.read_csv('E:\\data\\tick.csv', low_memory=False)
# working_path = 'E:\\data\\tick'
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
files_name = []
data = []
for root, dirs, files in os.walk(working_path):
    for fi in files:
        path = os.path.join(root, fi)
        files_name.append(path)
for i in files_name:
    if "20230424_20230504" in i:
        data.append(pd.read_feather(i))
data = pd.concat(data).reset_index(drop=True)


def handle_task(tick: pd.DataFrame, window_size, r_data):
    for _, g in tick.groupby('securityid'):
        groups = g.copy()
        groups['price_mean'] = ta.MA(groups.price, 20)  # 1分钟均价
        groups['price_mean5'] = ta.MA(groups.price, window_size)  # 5分钟均价
        groups['r_mean5'] = np.log(groups['price'] / groups['price_mean5'])
        groups['r_minute'] = ta.ROC(groups.price_mean, 20)
        groups['r_pre'] = groups['r_minute'].shift(-20)
        groups['r_real'] = ta.ROC(groups.price, 20)
        groups['r_real_pre'] = groups['r_real'].shift(-20)
        groups['mom'] = ta.MOM(groups.price_mean, window_size)
        groups['r_5'] = ta.ROC(groups.price_mean, window_size)

        """收益波动比"""
        groups['open5'] = groups['price'].shift(window_size - 1)
        groups['high5'] = ta.MAX(groups.high, window_size)
        groups['low5'] = ta.MIN(groups.low, window_size)
        groups['rwr'] = (groups['price'] - groups['open5']) / (groups['high5'] - groups['low5'])

        """波峰因子"""
        # groups['peaks'] = peak(groups, 20)

        """量价相关因子"""
        # groups['vc'] = cor_vc(groups, window_size)

        """买卖压力失衡因子"""
        # groups['voi'] = voi(groups)

        """峰度 偏度因子"""
        # groups['skew'] = cul_skew(groups['r_minute'], window_size)
        # groups['kurt'] = calculate_kurtosis(groups['r_minute'], 20)

        """最优波动率"""
        # groups['disaster'] = disaster(groups, window_size)

        """量价相关pearson"""
        groups['total_value_trade_ms'] = ta.SUM(groups['total_value_trade'], 20)
        groups['pearson'] = ta.CORREL(groups['total_value_trade_ms'], groups['price_mean'], window_size)

        """市场偏离度"""
        # groups['mpb'] = mpb(groups)

        """积极买入"""
        # groups['pob'] = positive_ratio(groups, window_size)

        lock.acquire()
        r_data.append(groups)
        lock.release()


def tick_handle(tick, window_size):
    """数据预处理，调整涨跌停价格。调仓周期为1分钟，滚动周期为5分钟"""
    group_index = ['securityid', 'date', 'time']
    tick.drop(tick[tick['eq_trading_phase_code'] != 'T'].index, inplace=True)
    tick.drop(tick.columns[~tick.columns.isin(col)], axis=1, inplace=True)
    tick.loc[tick['offer_price1'] == 0, 'offer_price1'] = np.nan
    tick.loc[tick['bid_price1'] == 0, 'bid_price1'] = np.nan
    tick['price'] = (tick['offer_price1'] + tick['bid_price1']) / 2
    tick.loc[tick['price'].isna(), 'price'] = tick['last']
    tick['minutes'] = (tick['time'] / 100000).astype('int')
    tick.drop(tick[tick['minutes'] < 930].index, inplace=True)
    tick.sort_values(group_index, inplace=True)

    tick_threads = []
    r_data = []
    k = 1
    stk = tick['securityid'].drop_duplicates(keep='first')
    len_sto = len(stk)
    group_size = len_sto // k
    remainder = len_sto % k
    start_index = 0
    for i in range(k):
        if i >= len_sto:
            break
        if i < remainder:
            end_index = start_index + group_size + 1
        else:
            end_index = start_index + group_size
        if end_index == len_sto:
            g = tick.loc[stk.index[start_index]:]
        else:
            g = tick.loc[stk.index[start_index]:stk.index[end_index]].drop(stk.index[end_index])
        tick_thread = threading.Thread(target=handle_task, args=(g, window_size, r_data))
        tick_thread.start()
        tick_threads.append(tick_thread)
        start_index = end_index

    for tick_thread in tick_threads:
        tick_thread.join()

    del tick
    rolling_data = pd.concat(r_data)
    del r_data
    rolling_data = pd.concat([rolling_data, fac_neutral(rolling_data, glob_f)], axis=1)
    return rolling_data


def ic_task(rd, date, tis):
    fac = rd[factors].apply(lambda x: x.corr(rd['r_real_pre']))
    ric = rd[factors].apply(lambda x: x.rank().corr(rd['r_real_pre'].rank()))
    ics = pd.concat([fac, ric])
    ics['date'] = date
    ics['time'] = tis
    # lock.acquire()
    factor_ic.append(ics)
    # lock.release()


ws = 4*20  # 经过验证3（或4）分钟的IR、IC最大
start_time = time.process_time()
data_minute = tick_handle(data, ws)
del data
data_minute.sort_values(['securityid', 'date', 'time'], inplace=True)
factors = glob_f
factors = [i + '_neutral' for i in factors]

# 计算IC/RankIC值
threads = []
ic = [i + '_ic' for i in factors]
rank_ic = [i + '_rank_ic' for i in factors]
for (da, ti), group in data_minute.groupby(['date', 'time']):
    ic_task(group, da, ti)
# for (da, ti), group in data_minute.groupby(['date', 'time']):
#     thread = threading.Thread(target=ic_task, args=(group, da, ti))
#     thread.start()
#     threads.append(thread)
#
# for thread in threads:
#     thread.join()

data_IC = pd.concat(factor_ic, axis=1).transpose()
data_IC.columns = ic + rank_ic + ['date', 'time']
data_IC = data_IC.sort_values(['date', 'time'])
del factor_ic

# 计算ICIR/RankICIR值
factor_ir = []
ir = [i + '_ir' for i in factors]
rank_ir = [i + '_rank_ir' for i in factors]
for _, group in data_IC.groupby('date'):
    group[ir] = group.rolling(window=ws)[ic].apply(lambda x: x.mean() / x.std())
    group[rank_ir] = group.rolling(window=ws)[rank_ic].apply(lambda x: x.mean() / x.std() if x.std() != 0 else 10)
    factor_ir.append(group)
data_IC = pd.concat(factor_ir)
del factor_ir


print(data_IC[rank_ic].mean())
# sto.cumsum().plot().set_xticks([])
# plt.show()
# data_IC.groupby('time')[ic].mean().plot.bar().set_xticks([])
# plt.show()
end_time = time.process_time()
print(end_time - start_time)