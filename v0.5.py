# import matplotlib.pyplot as plt
import os

import pandas as pd

from cul_funs import *
import threading
import time

# glob_f = ['voi', 'rwr', 'peaks', 'vc', 'skew', 'kurt', 'disaster', 'pearson', 'mpb', 'pob']
glob_f = ['pearson', 'rwr']
bao = []
for i in range(1, 6):
    bao.append('offer_price' + str(i))
    bao.append('offer_volume' + str(i))
    bao.append('bid_price' + str(i))
    bao.append('bid_volume' + str(i))
col = ['securityid', 'date', 'time', 'high', 'low', 'last', 'total_value_trade',
       'total_volume_trade', 'num_trades', 'dsmv'] + bao

lock = threading.Lock()

# data = pd.read_csv('E:\\data\\tick.csv', low_memory=False)
# working_path = 'E:\\data\\tick'
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
dsm = pd.read_csv('/Users/lvfreud/Desktop/中信建投/因子/data/TRD_Dalyr.csv')
dsm['dsmv'] = np.log(dsm['Dsmvtll'] / 100000.0)
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
data['securityid'] = data.securityid.astype('int')
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
dsm['Trddt'] = pd.to_datetime(dsm['Trddt'])
data = data.merge(dsm, left_on=['securityid', 'date'], right_on=['Stkcd', 'Trddt'], how='left')


def handle_task(tick: pd.DataFrame, window_size, r_data):
    """多线程函数"""
    for _, g in tick.groupby('securityid'):
        price_mean = ta.MA(g.price, 20)  # 1分钟均价
        price_mean5 = ta.MA(g.price, window_size)  # 5分钟均价
        r_mean5 = np.log(g['price'] / price_mean5)
        r_minute = ta.ROC(g.price, 20)
        r_pre = r_minute.shift(-20)
        # mom = ta.MOM(price_mean, window_size)
        r_5 = ta.ROC(g.price, window_size)
        groups = pd.concat([r_minute, r_5, r_mean5, r_pre], axis=1)
        groups.columns = ['r_minute', 'r_5', 'r_mean5', 'r_pre']

        """收益波动比"""
        open5 = g.price.shift(window_size - 1)
        high5 = ta.MAX(g.high, window_size)
        low5 = ta.MIN(g.low, window_size)
        groups['rwr'] = (g.price - open5) / (high5 - low5)

        """量价相关pearson"""
        total_value_trade_ms = g['total_value_trade']
        groups['pearson'] = ta.CORREL(total_value_trade_ms, g.price, window_size)

        """买卖压力失衡因子"""
        # groups['voi'] = voi(g)
        # groups['voi2'] = voi2(g)
        # groups['mofi'] = mofi(g)
        # groups['ori'] = ori(g)
        # groups['sori'] = sori(g)
        # groups['pir'] = pir(g)
        # groups['rsj'] = rsj(r_minute, window_size)
        # groups['illiq'] = illiq(g, r_minute)
        # groups['lsilliq'] = lsilliq(g, r_minute, window_size)
        # groups['gamma'] = gam(g, r_minute)
        # groups['lambda'] = lam(g, r_minute, window_size)
        # groups['lqs'] = lqs(g)

        """波峰因子"""
        # groups['peaks'] = peak(g, 20)

        """量价相关因子"""
        # groups['vc'] = cor_vc(g, window_size)

        """峰度 偏度因子"""
        # groups['skew'] = cul_skew(g['price'], window_size)
        # groups['kurt'] = r_minute.kurt()

        """最优波动率"""
        # groups['disaster'] = disaster(groups, window_size)

        """市场偏离度"""
        # groups['mpb'] = mpb(g)
        # groups['mpc'] = mpc(g)
        # groups['mpc_max'] = mpc_max(g)
        # groups['mpc_skew'] = mpc_skew(g)
        # groups['mcib'] = mci_b(g)
        # groups['ptor'] = ptor(g, r_minute)
        # groups['bni'] = bni(g, r_minute, window_size)
        # groups['mb'] = mb(g, window_size)
        # groups['bam'] = bam(g, 20)
        # groups['ba_cov'] = ba_cov(g, window_size)
        # groups['por'] = por(g, window_size)

        lock.acquire()
        r_data.append(groups)
        lock.release()


def tick_handle(tick: pd.DataFrame, window_size):
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
    tick['const'] = 1
    tick.sort_values(group_index, inplace=True)

    # 多线程计算，cores为线程数，一般为设置为cpu核心数，x86架构下可以提升运算速度
    tick_threads = []
    r_data = []
    cores = 1
    sis = tick['securityid'].drop_duplicates(keep='first')
    len_sto = len(sis)
    gs = len_sto // cores
    rmr = len_sto % cores
    sti = 0
    for tc in range(cores):  # 按线程数将数据分组
        if tc >= len_sto:
            break
        if tc < rmr:
            edi = sti + gs + 1
        else:
            edi = sti + gs
        if edi == len_sto:
            g = tick.loc[sis.index[sti]:]
        else:
            g = tick.loc[sis.index[sti]:sis.index[edi]].drop(sis.index[edi])
        tick_thread = threading.Thread(target=handle_task, args=(g, window_size, r_data))
        tick_thread.start()
        tick_threads.append(tick_thread)
        sti = edi

    for tick_thread in tick_threads:
        tick_thread.join()

    r_data = pd.concat(r_data, axis=0)
    tick = pd.concat([tick, r_data], axis=1, copy=False)
    tick = pd.concat([tick, fac_neutral(tick, glob_f)], axis=1, copy=False)  # 中性化，多线程建议使用fac_neutral2
    return tick


ws = 5*20
start_time = time.process_time()
data = tick_handle(data, ws)
data.sort_values(['securityid', 'date', 'time'], inplace=True)
factors = glob_f
factors = [i + '_neutral' for i in factors]

# 计算IC/RankIC值
factor_ic = []
ic = [i + '_ic' for i in factors]
rank_ic = [i + '_rank_ic' for i in factors]
for (da, ti), group in data.groupby(['date', 'time']):
    fac = group[['r_pre']+factors].corr().iloc[0, 1:].to_list()
    ric = group[['r_pre']+factors].rank().corr().iloc[0, 1:].to_list()
    factor_ic.append([da, ti] + fac + ric)
data_IC = pd.DataFrame(factor_ic, columns=['date', 'time'] + ic + rank_ic)
data_IC = data_IC.sort_values(['date', 'time'])
print(data_IC[rank_ic].mean())
del factor_ic

# 计算ICIR/RankICIR值
data_ir = data_IC[ic].mean() / data_IC[ic].std()
data_rank_ir = data_IC[rank_ic].mean() / data_IC[rank_ic].std()
print(data_rank_ir)

# 分组回测
for kk in factors:
    sto = returns_stock(data, kk)
    print(kk)
    print(sto.sum()/len(data_IC['date'].drop_duplicates()))

# 绘图
# sto.cumsum().plot().set_xticks([])
# plt.show()
# data_IC.groupby('time')[ic].mean().plot.bar().set_xticks([])
# plt.show()
end_time = time.process_time()
print(end_time - start_time)