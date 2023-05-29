import talib as ta
import numpy as np
import pandas as pd
import threading

lock = threading.Lock()


def cul_skew(returns, window_size):
    """偏度计算"""
    stddev = ta.STDDEV(returns, window_size)
    skew = (returns - ta.MA(returns, window_size)) / stddev
    skew[skew.isna()] = 0
    skew[skew == np.inf] = 0
    skew = ta.SMA(skew ** 3, window_size)
    return skew


def calculate_kurtosis(returns, window_size):
    """峰度计算"""
    sma = ta.SMA(returns, window_size)
    stddev = ta.STDDEV(returns, window_size)
    kurt = (returns - sma) / stddev
    kurt[kurt.isna()] = 0
    kurt[kurt == np.inf] = 0
    kurt = ta.MA(kurt ** 4, window_size)
    return kurt


def calculate_residuals(x, y) -> np.array:
    """残差计算"""
    ones = np.ones((x.shape[0], 1))
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x = np.concatenate((ones, x), axis=1)
    try:
        beta = np.linalg.inv(x.T @ x) @ x.T @ y
        y_pred = x @ beta
        res = y - y_pred
    except np.linalg.LinAlgError:  # 如果获得奇异矩阵，则说明残差为0
        res = 0
    return res


def fac_neutral(rolling_data: pd.DataFrame, factor_origin) -> pd.DataFrame:
    """中性化"""
    rolling_residuals = []
    fac_name = [i + '_neutral' for i in factor_origin]
    for (_, _), g in rolling_data.groupby(['date', 'time']):
        neu = []
        x = g[['r_minute', 'r_5', 'r_mean5']].values
        for i in factor_origin:
            neu.append(calculate_residuals(x, g[i].values))
        neu = pd.DataFrame(neu, columns=g.index, index=fac_name)
        rolling_residuals.append(neu.T)
    rolling_residuals = pd.concat(rolling_residuals)
    rolling_residuals.fillna(0, inplace=True)
    return rolling_residuals


def cul_res(rolling_data, factor_origin, neu):
    fac_name = [i + '_neutral' for i in factor_origin]
    rrs = []
    for (_, _), g in rolling_data.groupby(['date', 'time']):
        ne = []
        x = g[['r_minute', 'r_5', 'r_mean5']].values
        for i in factor_origin:
            ne.append(calculate_residuals(x, g[i].values))
        ne = pd.DataFrame(ne, columns=g.index, index=fac_name)
        rrs.append(ne.T)
    rrs = pd.concat(rrs)
    lock.acquire()
    neu.append(rrs)
    lock.release()


def fac_neutral2(rolling_data: pd.DataFrame, factor_origin):
    """中性化"""
    threads = []
    neu = []
    k = 16
    rolling_data.sort_values(['date', 'time'], inplace=True)
    stk = rolling_data[['date', 'time']].drop_duplicates(keep='first')
    dt_len = len(stk)
    group_size = dt_len // k
    remainder = dt_len % k
    start_index = 0
    for i in range(k):
        if i >= dt_len:
            break
        if i < remainder:
            end_index = start_index + group_size + 1
        else:
            end_index = start_index + group_size
        if end_index == dt_len:
            g = rolling_data.loc[stk.index[start_index]:]
        else:
            g = rolling_data.loc[stk.index[start_index]:stk.index[end_index]].drop(stk.index[end_index])
        tick_thread = threading.Thread(target=cul_res, args=(g, factor_origin, neu))
        tick_thread.start()
        threads.append(tick_thread)
        start_index = end_index

    for thread in threads:
        thread.join()
    rolling_residuals = pd.concat(neu)
    rolling_residuals.fillna(0, inplace=True)
    return rolling_residuals


def voi(data_dic: pd.DataFrame):
    """voi订单失衡 Volume Order Imbalance20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    tvt = ta.SUM(data_dic['total_volume_trade'], ticks_num)
    bid_sub_price = data_dic['bid_price1'] - data_dic['bid_price1'].shift(ticks_num)
    ask_sub_price = data_dic['offer_price1'] - data_dic['offer_price1'].shift(ticks_num)
    bid_sub_volume = data_dic['bid_volume1'] - data_dic['bid_volume1'].shift(ticks_num)
    ask_sub_volume = data_dic['offer_volume1'] - data_dic['offer_volume1'].shift(ticks_num)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = data_dic['bid_volume1'][bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = data_dic['offer_volume1'][ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / tvt
    return tick_fac_data


def voi2(data_dic: pd.DataFrame):
    """多档voi订单失衡 Volume Order Imbalance 20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    tvt = ta.SUM(data_dic['total_volume_trade'], ticks_num)
    weighted_bp = data_dic['bid_price1']
    weighted_op = data_dic['offer_price1']
    weighted_bv = data_dic['bid_volume1']
    weighted_ov = data_dic['offer_volume1']
    w = [(1-(i-1)/5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(2, 6):
        weighted_bp += data_dic['bid_price' + str(i)] * w[i - 1]
        weighted_op += data_dic['offer_price' + str(i)] * w[i - 1]
        weighted_bv += data_dic['bid_volume' + str(i)] * w[i - 1]
        weighted_ov += data_dic['offer_volume' + str(i)] * w[i - 1]
    weighted_bp = weighted_bp / ws
    weighted_op = weighted_op / ws
    weighted_bv = weighted_bv / ws
    weighted_ov = weighted_ov / ws
    bid_sub_price = weighted_bp - weighted_bp.shift(ticks_num)
    ask_sub_price = weighted_op - weighted_op.shift(ticks_num)
    bid_sub_volume = weighted_bv - weighted_bv.shift(ticks_num)
    ask_sub_volume = weighted_ov - weighted_ov.shift(ticks_num)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = weighted_bv[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = weighted_ov[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / tvt
    return tick_fac_data


def mofi(data: pd.DataFrame):
    """改进voi 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    tvt = ta.SUM(data['total_volume_trade'], ticks_num)
    tick_fac_data = []
    w = [i / 5 for i in range(1, 6)]
    for i in range(1, 6):
        bid_sub_price = data['bid_price' + str(i)] - data['bid_price' + str(i)].shift(ticks_num)
        ask_sub_price = data['offer_price' + str(i)] - data['offer_price' + str(i)].shift(ticks_num)
        bid_sub_volume = data['bid_volume' + str(i)] - data['bid_volume' + str(i)].shift(ticks_num)
        ask_sub_volume = data['offer_volume' + str(i)] - data['offer_volume' + str(i)].shift(ticks_num)
        bid_volume_change = bid_sub_volume
        ask_volume_change = ask_sub_volume
        # bid_volume_change[bid_sub_price == 0] = bid_sub_volume[bid_sub_price == 0]
        bid_volume_change[bid_sub_price < 0] = -data['bid_volume' + str(i)].shift(ticks_num)[bid_sub_price < 0]
        bid_volume_change[bid_sub_price > 0] = data['bid_volume' + str(i)][bid_sub_price > 0]
        ask_volume_change[ask_sub_price < 0] = data['offer_volume' + str(i)][ask_sub_price < 0]
        ask_volume_change[ask_sub_price > 0] = -data['offer_volume' + str(i)].shift(ticks_num)[ask_sub_price > 0]
        if i == 1:
            tick_fac_data = (bid_volume_change - ask_volume_change) * w[i-1]
        else:
            tick_fac_data += (bid_volume_change - ask_volume_change) * w[i - 1]
    tick_fac_data = tick_fac_data / sum(w) / tvt
    return tick_fac_data


def ori(data_dic: pd.DataFrame):
    """ Order Imbalance Ratio 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    weighted_bv = data_dic['bid_volume1']
    weighted_ov = data_dic['offer_volume1']
    w = [(1-(i-1)/5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(2, 6):
        weighted_bv += data_dic['bid_volume' + str(i)] * w[i - 1]
        weighted_ov += data_dic['offer_volume' + str(i)] * w[i - 1]
    weighted_bv = weighted_bv / ws
    weighted_ov = weighted_ov / ws
    tick_fac_data = (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov)
    tick_fac_data = ta.SMA(tick_fac_data, ticks_num)
    return tick_fac_data


def sori(data_dic: pd.DataFrame):
    """改进加权 Order Imbalance Ratio 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    w = [(1-(i-1)/5) for i in range(1, 6)]
    tick_fac_data = []
    for i in range(1, 6):
        weighted_bv = data_dic['bid_volume' + str(i)]
        weighted_ov = data_dic['offer_volume' + str(i)]
        if i == 1:
            tick_fac_data = (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov) * w[i - 1]
        else:
            tick_fac_data += (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov) * w[i - 1]
    tick_fac_data = tick_fac_data / sum(w)
    tick_fac_data = ta.SMA(tick_fac_data, ticks_num)
    return tick_fac_data


def pir(data_dic: pd.DataFrame):
    """pir Price Imbalance Ratio 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    weighted_bv = data_dic['bid_price1']
    weighted_ov = data_dic['offer_price1']
    w = [(1 - (i - 1) / 5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(2, 6):
        weighted_bv += data_dic['bid_price' + str(i)] * w[i - 1]
        weighted_ov += data_dic['offer_price' + str(i)] * w[i - 1]
    weighted_bv = weighted_bv / ws
    weighted_ov = weighted_ov / ws
    tick_fac_data = (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov)
    tick_fac_data = ta.SMA(tick_fac_data, ticks_num)
    return tick_fac_data


def rsj(returns, window_size):
    """Relative Signed Jump"""
    returns2 = returns ** 2
    rv = ta.SUM(returns2, window_size)
    rv_up = ta.SUM(returns2[returns > 0], window_size)
    rv_down = ta.SUM(returns2[returns < 0], window_size)
    tick_fac = (rv_up - rv_down) / rv
    return tick_fac


def lam(data_dic: pd.DataFrame, window_size):
    """带方向的成交额对收益率的影响"""
    sign_vol = ta.SMA(data_dic['total_volume_trade'], 20) / 10000
    sign_vol[data_dic['r_minute'] < 0] = -sign_vol[data_dic['r_minute'] < 0]
    la = ta.BETA(data_dic['r_minute'], sign_vol, window_size)
    return la


def lqs(data_dic: pd.DataFrame):
    """Log quote Slope 因子 描述盘口的形状"""
    bp1 = ta.SMA(data_dic['bid_price1'], 20)
    of1 = ta.SMA(data_dic['offer_price1'], 20)
    bv1 = ta.SMA(data_dic['bid_volume1'], 20)
    ov1 = ta.SMA(data_dic['offer_volume1'], 20)
    tick_fac = (ta.LN(bp1) - ta.LN(of1)) / (ta.LN(bv1) + ta.LN(ov1))
    return tick_fac


def peak(data_dic: pd.DataFrame, window_size: int):
    """波峰因子"""
    vol_mean = ta.MA(data_dic.total_volume_trade, window_size)
    vol_std = ta.STDDEV(data_dic.total_volume_trade, window_size)
    data_dic.loc[data_dic['total_volume_trade'] > vol_mean+vol_std, 'peak'] = 1
    data_dic['peak'].fillna(0, inplace=True)
    return ta.SUM(data_dic.peak, window_size)


def cor_vc(data_dic: pd.DataFrame, window_size):
    """量价相关因子"""
    minute_trade = ta.SUM(data_dic.total_volume_trade, 20)  # 分钟交易量
    dav = (data_dic['price_mean'] - data_dic['price_mean5'])*minute_trade
    vol_std = ta.STDDEV(data_dic.total_volume_trade, 20)
    last_std = ta.STDDEV(data_dic['last'], 20)
    vc = ta.SUM(dav, window_size)/(vol_std*last_std)
    vc[vc == np.inf] = np.nan
    vc[vc == 0] = np.nan
    return vc


def disaster(minute_data: pd.DataFrame, window_size):
    """最优波动率"""
    results = []
    for i in range(window_size, len(minute_data)):
        window_data = minute_data.iloc[i - window_size:i][['price', 'last', 'high', 'low']].values.flatten()
        ratio_squared = (window_data.std() / window_data.mean()) ** 2
        results.append(ratio_squared)
    ratio = np.array([np.nan] * window_size + results)
    ratio = np.where(ratio == 0, np.nan, ratio)
    ratio = minute_data['r_minute'].values/ratio
    return ratio


def mpb(data_dic):
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    va_t = ta.SUM(data_dic['total_value_trade'], ticks_num)
    vol_t = ta.SUM(data_dic['total_volume_trade'], ticks_num)
    tp = va_t / vol_t  # 注意单位
    tp[vol_t == 0] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = data_dic['price_mean']
    tick_fac_data = tp - (mid + mid.shift(ticks_num)) / 10000 / 2
    return tick_fac_data


def mpc(data_dic):
    """Midpoint Price Change 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    mid = (ta.SMA(data_dic['bid_price1'], ticks_num) + ta.SMA(data_dic['offer_price1'], ticks_num)) / 2
    tick_fac_data = ta.ROCP(mid, 100)
    return tick_fac_data


def mci_b(data_dic: pd.DataFrame):
    """Marginal Cost of Immediacy 边际交易成本 中信建投-高频选股因子分类体系"""
    mid = data_dic['price_mean'] / 10000
    dol_vol_b = data_dic['bid_price1'] * data_dic['bid_volume1'] / 10000
    q_bid = data_dic['bid_volume1']
    for i in range(2, 6):
        dol_vol_b += data_dic['bid_price' + str(i)] * data_dic['bid_volume' + str(i)] / 10000
        q_bid += data_dic['bid_volume' + str(i)]
    v_wap = - (dol_vol_b / q_bid - mid) / mid * 10000  # 单位：价格变动bp
    tick_fac = v_wap / dol_vol_b * 10000  # 单位：bp/万元
    return tick_fac


def ptor(data_dic: pd.DataFrame):
    ticks_num = 20
    amt = ta.SMA(data_dic['total_value_trade'], ticks_num)
    tv = ta.SMA(data_dic['num_trades'], ticks_num)
    amt_per_trade = amt / tv
    apt_out = ta.SUM(amt[data_dic['r_minute'] < 0], ticks_num) / ta.SUM(tv[data_dic['r_minute'] < 0], ticks_num)
    tick_fac = apt_out / amt_per_trade
    return tick_fac


def bni(data_dic: pd.DataFrame, window_size):
    """BNI 大资金流入比例"""
    ticks_num = 20
    amt = ta.SUM(data_dic['total_value_trade'], ticks_num)
    tv = ta.SUM(data_dic['num_trades'], ticks_num)
    apt = amt / tv
    tick_fac = []
    for i in range(window_size, len(data_dic)):
        apt_r = apt.iloc[i-window_size:i]
        apt_big = apt_r.rank(ascending=False)
        big_rank = 0.3 * len(apt_r)
        big_net_in = amt[(apt_big < big_rank)&(data_dic['r_minute'] > 0)].sum() - amt[(apt_big < big_rank)&(data_dic['r_minute'] > 0)].sum()
        big_net_per = big_net_in / tv.iloc[i]
        tick_fac.append(big_net_per / apt.iloc[i])
    tick_fac = [np.nan] * window_size + tick_fac
    return tick_fac


def mb(data_dic: pd.DataFrame, window_size):
    """MB 大资金驱动涨幅"""
    ticks_num = 20
    amt = ta.SUM(data_dic['total_value_trade'], ticks_num)
    tv = ta.SUM(data_dic['num_trades'], ticks_num)
    apt = amt / tv
    tick_fac = []
    mbs = 1
    for i in range(window_size, len(data_dic)):
        apt_r = apt.iloc[i-window_size:i]
        apt_big = apt_r.rank(ascending=False)
        big_rank = 0.3 * len(apt_r)
        mbs *= data_dic['r_minute'][apt_big < big_rank] / 20 + 1
        tick_fac.append(mbs)
    tick_fac = [np.nan] * window_size + tick_fac
    return tick_fac


def bam(data_dic: pd.DataFrame, window_size):
    buy_positive = pd.DataFrame(0, columns=['t1', 't2', 't3'], index=data_dic.index)
    tvt = ta.SUM(data_dic['total_value_trade'], window_size)
    trade1 = data_dic['total_value_trade'][data_dic['bid_price1'].shift(1) <= data_dic['last']]
    trade2 = data_dic['total_value_trade'][data_dic['offer_price1'].shift(1) >= data_dic['last']]
    trade3 = data_dic['total_value_trade'][~((trade1 > 0) & (trade2 > 0))] / 2
    buy_positive.loc[data_dic['last'] >= data_dic['bid_price1'].shift(1), 't1'] = trade1
    buy_positive.loc[data_dic['last'] <= data_dic['offer_price1'].shift(1), 't2'] = trade2
    buy_positive.loc[data_dic[~((trade1 > 0) & (trade2 > 0))], 't3'] = trade3
    bam_t = ta.SUM(buy_positive['t1']+buy_positive['t3'], window_size) / tvt
    sam_t = ta.MA(buy_positive['t2']+buy_positive['t3'], window_size) / tvt
    return bam_t, sam_t


def ba_cov(data_dic: pd.DataFrame, window_size):
    buy_positive = pd.DataFrame(0, columns=['t1', 't2', 't3'], index=data_dic.index)
    trade1 = data_dic['total_value_trade'][data_dic['bid_price1'].shift(1) <= data_dic['last']]
    trade2 = data_dic['total_value_trade'][data_dic['offer_price1'].shift(1) >= data_dic['last']]
    trade3 = data_dic['total_value_trade'][~((trade1 > 0) & (trade2 > 0))] / 2
    buy_positive.loc[data_dic['last'] >= data_dic['bid_price1'].shift(1), 't1'] = trade1
    buy_positive.loc[data_dic['last'] <= data_dic['offer_price1'].shift(1), 't2'] = trade2
    buy_positive.loc[data_dic[~((trade1 > 0) & (trade2 > 0))], 't3'] = trade3
    ba = buy_positive['t1']+buy_positive['t3']
    sa = buy_positive['t2']+buy_positive['t3']
    ba = ta.MA(ba, window_size) / ta.STDDEV(ba, window_size)
    sa = ta.MA(sa, window_size) / ta.STDDEV(sa, window_size)
    return ba, sa


def positive_ratio(data_dic, tick_nums):
    """积极买入成交额占总成交额的比例"""
    buy_positive = pd.DataFrame(0, columns=['total_value_trade'], index=data_dic.index)
    buy_positive.loc[data_dic['last'] >= data_dic['offer_price1'].shift(1), 'total_value_trade'] = \
        data_dic['total_value_trade'][data_dic['last'] >= data_dic['offer_price1'].shift(1)]
    tick_fac_data = ta.SUM(buy_positive['total_value_trade'], tick_nums) / \
                    ta.SUM(data_dic['total_value_trade'], tick_nums)
    return tick_fac_data


def por(data_dic, tick_nums):
    """积极买入成交额占总成交额的比例"""
    buy_positive = pd.DataFrame(0, columns=['total_value_trade'], index=data_dic.index)
    op = ta.MA(data_dic['offer_price1'], 20)
    last = ta.MA(data_dic['last'], 20)
    tvt = ta.MA(data_dic['total_value_trade'], 20)
    buy_positive.loc[last >= op.shift(20), 'total_value_trade'] = tvt[last >= op.shift(20)]
    tick_fac_data = ta.SUM(buy_positive['total_value_trade'], tick_nums) / ta.SUM(tvt, tick_nums)
    return tick_fac_data


def returns_stock(data: pd.DataFrame, factor) -> pd.DataFrame:
    sto = []
    k = 10
    for (da, mi), group in data.groupby(['date', 'minutes']):
        stk = group.groupby('securityid')[factor].mean().reset_index()
        stk['date'] = da
        stk['minutes'] = mi
        pre = group.drop_duplicates(subset='securityid', keep='last')[['securityid', 'date', 'minutes', 'r_pre']]
        stk = stk.merge(pre, on=['securityid', 'date', 'minutes'], how='left').sort_values(factor, ascending=False)
        group_size = len(stk) // k
        remainder = len(stk) % k
        start_index = 0
        stocks = pd.DataFrame()
        for i in range(k):
            if i < remainder:
                end_index = start_index + group_size + 1
            else:
                end_index = start_index + group_size
            stocks['r_pre'+str(i)] = stk[start_index:end_index]['r_pre'].reset_index(drop=True)
            start_index = end_index
        stocks['date'] = da
        stocks['minutes'] = mi
        sto.append(stocks)
    sto = pd.concat(sto).set_index(['date', 'minutes'])
    return sto
