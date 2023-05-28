import talib as ta
import numpy as np
import pandas as pd
import threading

lock = threading.Lock()


def cul_skew(returns, window_size):
    """偏度计算"""
    stddev = ta.STDDEV(returns['r_minute'], window_size)
    skew = (returns['r_minute'] - returns['r_mean5']) / stddev
    skew[skew.isna()] = 0
    skew[skew == np.inf] = 0
    skew = ta.SMA(skew, window_size)
    return skew


def calculate_kurtosis(returns, window_size):
    """峰度计算"""
    sma = ta.SMA(returns, window_size)
    stddev = ta.STDDEV(returns, window_size)
    kurt = ta.MA((returns - sma) ** 4, window_size)/(stddev**4)
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
    for (_, _), g in rolling_data.groupby(['date', 'time']):
        groups = pd.DataFrame(columns=fac_name, index=g.index)
        x = g[['r_minute', 'r_5', 'r_mean5']].values
        for i in factor_origin:
            groups[i + '_neutral'] = calculate_residuals(x, g[i].values)
        lock.acquire()
        neu.append(groups)
        lock.release()
    return


def fac_neutral2(rolling_data: pd.DataFrame, factor_origin):
    """中性化"""
    threads = []
    neu = []
    k = 1
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
    tp[np.isinf(tp)] = np.nan
    tp.fillna(method='ffill', inplace=True)
    mid = data_dic['price']
    tick_fac_data = tp - (mid + mid.shift(ticks_num)) / 10000 / 2
    tick_fac_data = ta.SUM(tick_fac_data, 80)
    return tick_fac_data


def positive_ratio(data_dic, tick_nums):
    """积极买入成交额占总成交额的比例"""
    buy_positive = pd.DataFrame(0, columns=['total_value_trade'], index=data_dic.index)
    buy_positive.loc[data_dic['last'] >= data_dic['offer_price1'].shift(1), 'total_value_trade'] = \
        data_dic['total_value_trade'][data_dic['last'] >= data_dic['offer_price1'].shift(1)]
    tick_fac_data = ta.SUM(buy_positive['total_value_trade'], tick_nums) / \
                    ta.SUM(data_dic['total_value_trade'], tick_nums)
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
