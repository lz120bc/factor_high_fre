import matplotlib.pyplot as plt
from cul_funs import *
import threading
import time

# glob_f = ['pearson', 'rwr', 'voi', 'voi2', 'mofi', 'ori', 'sori', 'pir', 'rsj', 'illiq', 'lsilliq',
#           'lambda', 'lqs', 'peaks', 'vc', 'skew', 'kurt', 'mpb', 'mpc', 'mpc_max', 'mpc_skew',
#           'mcib', 'ptor', 'bni', 'mb', 'bam', 'ba_cov', 'por']
glob_f = ['bam']
lock = threading.Lock()
# working_path = 'E:\\data\\tick'
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
data = pd.read_feather(working_path+'/tickda.feather')


def handle_task(tick: pd.DataFrame, window_size, r_data):
    """多线程函数"""
    for _, g in tick.groupby('securityid'):
        price_mean5 = ta.MA(g.price, window_size)  # 5分钟均价
        r_mean5 = np.log(g['price'] / price_mean5)
        r_minute = ta.ROC(g.price, 20)
        r_pre = r_minute.shift(-20)
        r_5 = ta.ROC(g.price, window_size)
        groups = pd.concat([r_minute, r_5, r_mean5, r_pre], axis=1)
        groups.columns = ['r_minute', 'r_5', 'r_mean5', 'r_pre']

        # open5 = g.price.shift(window_size - 1)
        # high5 = ta.MAX(g.high, window_size)
        # low5 = ta.MIN(g.low, window_size)
        # groups['rwr'] = (g.price - open5) / (high5 - low5)

        # total_value_trade_ms = g['total_value_trade']
        # groups['pearson'] = ta.CORREL(total_value_trade_ms, g.price, window_size)

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
        # groups['peaks'] = peak(g, 20)
        # groups['vc'] = cor_vc(g, window_size)
        # groups['skew'] = cul_skew(g['price'], window_size)
        # groups['kurt'] = calculate_kurtosis(g['price'], window_size)
        # groups['disaster'] = disaster(groups, window_size)
        # groups['mpb'] = mpb(g)
        # groups['mpc'] = mpc(g)
        # groups['mpc_max'] = mpc_max(g)
        # groups['mpc_skew'] = mpc_skew(g)
        # groups['mcib'] = mci_b(g)
        # groups['ptor'] = ptor(g, r_minute)
        # groups['bni'] = bni(g, r_minute, window_size)
        # groups['mb'] = mb(g, window_size)
        groups['bam'] = bam(g, 20)
        # groups['ba_cov'] = ba_cov(g, window_size)
        # groups['por'] = por(g, 20)

        lock.acquire()
        r_data.append(groups)
        lock.release()


def tick_handle(tick: pd.DataFrame, window_size):
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
    del r_data
    print("culculate done!\t use time:%.2fs" % (time.process_time()))
    tick = pd.concat([tick, fac_neutral(tick, glob_f)], axis=1, copy=False)  # 中性化，多线程建议使用fac_neutral2
    print("neutralize done!\t use time:%.2fs" % (time.process_time()))
    return tick


ws = 5*20
start_time = time.process_time()
data = tick_handle(data, ws)
data.sort_values(['securityid', 'date', 'time'], inplace=True)
fac = ['voi_neutral', 'sori_neutral', 'pearson_neutral', 'mpc_skew_neutral', 'bam_neutral', 'por_neutral']

# 输出标准化因子值
# dar = []
# for (date, time), group in data.groupby(['date', 'tick']):
#     g = pd.DataFrame(index=group.index)
#     for factor in fac:
#         g[factor + '_rank'] = group[factor].rank(ascending=False) / len(g)
#     dar.append(g)
# dar = pd.concat(dar, axis=0)
# data = pd.concat([data, dar], axis=1)
# del dar
# data.to_feather(working_path+'/tickf.feather')
factors = glob_f
factors = [i + '_neutral' for i in factors]

# 计算IC/RankIC值
factor_ic = []
ic = [i + '_ic' for i in factors]
rank_ic = [i + '_rank_ic' for i in factors]
for (da, ti), group in data.groupby(['date', 'tick']):
    fac = group[['r_pre']+factors].corr().iloc[0, 1:].to_list()
    ric = group[['r_pre']+factors].rank().corr().iloc[0, 1:].to_list()
    factor_ic.append([da, ti] + fac + ric)
data_IC = pd.DataFrame(factor_ic, columns=['date', 'tick'] + ic + rank_ic)
data_IC = data_IC.sort_values(['date', 'tick'])
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
    print(sto.sum() / len(data['date'].drop_duplicates()))
    sto.cumsum().plot().set_xticks([])
#     plt.savefig(kk + '.png')
#
# end_time = time.process_time()
# print(end_time - start_time)
