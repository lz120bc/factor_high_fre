import talib as ta
import numpy as np
import pandas as pd
import threading
import statsmodels.api as sm
from scipy.ndimage import shift

# 计算函数
lock = threading.Lock()


def cul_skew(returns, window_size) -> pd.Series:
    """偏度计算"""
    stddev = ta.STDDEV(returns, window_size)
    skew = (returns - ta.MA(returns, window_size)) / stddev
    skew[skew.isna()] = 0
    skew[np.isinf(skew)] = 0
    skew = skew ** 3
    return skew


def calculate_kurtosis(returns, window_size) -> pd.Series:
    """峰度计算"""
    sma = ta.SMA(returns, window_size)
    stddev = ta.STDDEV(returns, window_size)
    kurt = (returns - sma) / stddev
    kurt[kurt.isna()] = 0
    kurt[np.isinf(kurt)] = 0
    kurt = kurt ** 4
    return kurt


def calculate_residuals(x, y) -> np.array:
    """残差计算"""
    try:
        beta = np.linalg.inv(x.T @ x) @ x.T @ y
        beta = np.where(np.isinf(beta), np.nan, beta)
        y_pred = x @ beta
        res = y - y_pred
    except np.linalg.LinAlgError:  # 如果获得奇异矩阵，则说明残差为0
        res = 0
    return res


def calculate_residuals2(x, y) -> np.array:
    """残差计算"""
    ols = sm.OLS(y, x).fit()
    res = y - ols.predict(x)
    return res


def cb(x, y) -> np.array:
    """beta计算"""
    try:
        beta = np.linalg.inv(x.T @ x) @ x.T @ y
        res = beta[1]
    except np.linalg.LinAlgError:  # 如果获得奇异矩阵，则说明残差为0
        res = 0
    return res


def fac_neutral(rolling_data: pd.DataFrame, factor_origin) -> pd.DataFrame:
    """中性化"""
    rolling_residuals = []
    const = ['r_minute', 'r_5', 'r_mean5', 'const', 'dsmv']
    fac_name = [i + '_neutral' for i in factor_origin]
    for (_, _), g in rolling_data.groupby(['date', 'tick']):
        neu = []
        for i in factor_origin:
            x = g[const + [i]].dropna()
            if len(x) == 0:
                neu.append(pd.Series(np.nan))
                continue
            cr = calculate_residuals(x[const].values, x[i].values)
            cr = pd.Series(cr, index=x.index)
            neu.append(cr)
        neu = pd.concat(neu, axis=1)
        neu.columns = fac_name
        if neu.isna().all().all():
            continue
        rolling_residuals.append(neu)
    rolling_residuals = pd.concat(rolling_residuals, axis=0)
    rolling_residuals.fillna(0, inplace=True)
    return rolling_residuals


def cul_res(rolling_data, factor_origin, neu) -> None:
    fac_name = [i + '_neutral' for i in factor_origin]
    rrs = []
    for (_, _), g in rolling_data.groupby(['date', 'time']):
        ne = []
        x = g[['r_minute', 'r_5', 'r_mean5', 'const', 'dsmv']]
        for i in factor_origin:
            ne.append(calculate_residuals(x, g[i]))
        ne = pd.DataFrame(ne, columns=g.index, index=fac_name)
        rrs.append(ne.T)
    rrs = pd.concat(rrs)
    lock.acquire()
    neu.append(rrs)
    lock.release()


def fac_neutral2(rolling_data: pd.DataFrame, factor_origin):
    """多线程中性化"""
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


def voi(data_dic: pd.DataFrame) -> pd.Series:
    """voi订单失衡 Volume Order Imbalance20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    bid_sub_price = data_dic['bid_price1'] - data_dic['bid_price1'].shift(ticks_num)
    ask_sub_price = data_dic['offer_price1'] - data_dic['offer_price1'].shift(ticks_num)
    bid_sub_volume = data_dic['bid_volume1'] - data_dic['bid_volume1'].shift(ticks_num)
    ask_sub_volume = data_dic['offer_volume1'] - data_dic['offer_volume1'].shift(ticks_num)
    bid_volume_change = bid_sub_volume
    ask_volume_change = ask_sub_volume
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = data_dic['bid_volume1'][bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = data_dic['offer_volume1'][ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / 100000
    return tick_fac_data


def voi2(data_dic: pd.DataFrame) -> pd.Series:
    """多档voi订单失衡 Volume Order Imbalance 20200709-中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    weighted_bp = pd.Series(0, index=data_dic.index)
    weighted_op = pd.Series(0, index=data_dic.index)
    weighted_bv = pd.Series(0, index=data_dic.index)
    weighted_ov = pd.Series(0, index=data_dic.index)
    w = [(1 - (i - 1) / 5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(1, 6):
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
    bid_volume_change = weighted_bv - weighted_bv.shift(ticks_num)
    ask_volume_change = weighted_ov - weighted_ov.shift(ticks_num)
    bid_volume_change[bid_sub_price < 0] = 0
    bid_volume_change[bid_sub_price > 0] = weighted_bv[bid_sub_price > 0]
    ask_volume_change[ask_sub_price < 0] = weighted_ov[ask_sub_price < 0]
    ask_volume_change[ask_sub_price > 0] = 0
    tick_fac_data = (bid_volume_change - ask_volume_change) / 100000
    return tick_fac_data


def mofi(data: pd.DataFrame) -> pd.Series:
    """改进voi 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    tick_fac_data = pd.Series(0, index=data.index)
    w = [i / 5 for i in range(1, 6)]
    for i in range(1, 6):
        bid_sub_price = data['bid_price' + str(i)] - data['bid_price' + str(i)].shift(ticks_num)
        ask_sub_price = data['offer_price' + str(i)] - data['offer_price' + str(i)].shift(ticks_num)
        bid_volume_change = data['bid_volume' + str(i)] - data['bid_volume' + str(i)].shift(ticks_num)
        ask_volume_change = data['offer_volume' + str(i)] - data['offer_volume' + str(i)].shift(ticks_num)
        bid_volume_change[bid_sub_price < 0] = -data['bid_volume' + str(i)].shift(ticks_num)[bid_sub_price < 0]
        bid_volume_change[bid_sub_price > 0] = data['bid_volume' + str(i)][bid_sub_price > 0]
        ask_volume_change[ask_sub_price < 0] = data['offer_volume' + str(i)][ask_sub_price < 0]
        ask_volume_change[ask_sub_price > 0] = -data['offer_volume' + str(i)].shift(ticks_num)[ask_sub_price > 0]
        tick_fac_data += (bid_volume_change - ask_volume_change) * w[i - 1]
    tick_fac_data = tick_fac_data / sum(w) / 100000
    return tick_fac_data


def ori(data_dic: pd.DataFrame) -> pd.Series:
    """ Order Imbalance Ratio 中信建投-高频选股因子分类体系"""
    weighted_bv = pd.Series(0, index=data_dic.index)
    weighted_ov = pd.Series(0, index=data_dic.index)
    w = [(1 - (i - 1) / 5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(1, 6):
        weighted_bv += data_dic['bid_volume' + str(i)] * w[i - 1]
        weighted_ov += data_dic['offer_volume' + str(i)] * w[i - 1]
    weighted_bv = weighted_bv / ws
    weighted_ov = weighted_ov / ws
    tick_fac_data = (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov)
    return tick_fac_data


def sori(data_dic: pd.DataFrame) -> pd.Series:
    """改进加权 Order Imbalance Ratio 中信建投-高频选股因子分类体系"""
    w = [(1 - (i - 1) / 5) for i in range(1, 6)]
    tick_fac_data = pd.Series(0, index=data_dic.index)
    for i in range(1, 6):
        weighted_bv = data_dic['bid_volume' + str(i)]
        weighted_ov = data_dic['offer_volume' + str(i)]
        tick_fac_data += (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov) * w[i - 1]
    tick_fac_data = tick_fac_data / sum(w)
    return tick_fac_data


def pir(data_dic: pd.DataFrame) -> pd.Series:
    """pir Price Imbalance Ratio 中信建投-高频选股因子分类体系"""
    weighted_bv = pd.Series(0, index=data_dic.index)
    weighted_ov = pd.Series(0, index=data_dic.index)
    w = [(1 - (i - 1) / 5) for i in range(1, 6)]
    ws = sum(w)
    for i in range(1, 6):
        weighted_bv += data_dic['bid_price' + str(i)] * w[i - 1]
        weighted_ov += data_dic['offer_price' + str(i)] * w[i - 1]
    weighted_bv = weighted_bv / ws
    weighted_ov = weighted_ov / ws
    tick_fac_data = (weighted_bv - weighted_ov) / (weighted_bv + weighted_ov)
    if tick_fac_data.isna().all():
        tick_fac_data = 0
    return tick_fac_data


def rsj(returns, window_size) -> pd.Series:
    """Relative Signed Jump"""
    returns2 = returns ** 2
    rv = ta.SUM(returns2, window_size)
    rv_up = pd.Series(0, index=returns.index)
    rv_down = pd.Series(0, index=returns.index)
    rv_up[returns > 0] = returns2[returns > 0]
    rv_down[returns < 0] = returns2[returns < 0]
    rv_up = ta.SUM(rv_up, window_size)
    rv_down = ta.SUM(rv_down, window_size)
    tick_fac = (rv_up - rv_down) / rv
    tick_fac[tick_fac == np.nan] = 0
    tick_fac[tick_fac == np.inf] = 0
    return tick_fac


def illiq(data_dic: pd.DataFrame, r_minute) -> pd.Series:
    liq = r_minute / data_dic['volumes'] * 1000000
    liq = abs(liq)
    liq = ta.SMA(liq, 20)
    return liq


def lsilliq(data_dic: pd.DataFrame, r_minute: pd.Series, window_size: int) -> pd.Series:
    liq = illiq(data_dic, r_minute)
    b = ta.BETA(abs(r_minute), data_dic['volumes'], window_size)
    cv = ta.STDDEV(data_dic['volumes'], window_size) / ta.SMA(data_dic['volumes'], window_size)
    cv.fillna(0, inplace=True)
    liq = liq - b * (cv ** 2)
    return liq


def gam(data_dic: pd.DataFrame, r_minute) -> pd.Series:
    sign_vol = data_dic['volumes'].copy()
    sign_vol[r_minute < 0] = -sign_vol[r_minute < 0]
    x = pd.concat([r_minute.shift(20), sign_vol, data_dic['const']], axis=1)
    ga = r_minute.rolling(window=100).apply(lambda y: cb(x.loc[y.index].values, y.values))
    return ga


def lam(data_dic: pd.DataFrame, r_minute, window_size) -> pd.Series:
    """带方向的成交额对收益率的影响"""
    sign_vol = ta.SMA(data_dic['volumes'], 20) / 10000
    sign_vol[r_minute < 0] = -sign_vol[r_minute < 0]
    la = ta.BETA(r_minute, sign_vol, window_size)
    return la


def lqs(data_dic: pd.DataFrame) -> pd.Series:
    """Log quote Slope 因子 描述盘口的形状"""
    bp1 = data_dic['bid_price1']
    of1 = data_dic['offer_price1']
    bv1 = data_dic['bid_volume1']
    ov1 = data_dic['offer_volume1']
    if bp1.isna().all() or of1.isna().all() or bv1.isna().all() or ov1.isna().all():
        tick_fac = 0
    else:
        tick_fac = (ta.LN(bp1) - ta.LN(of1)) / (ta.LN(bv1) + ta.LN(ov1))
        tick_fac[tick_fac.isna()] = 0
        tick_fac = tick_fac * 10000
    return tick_fac


def peak(data_dic: pd.DataFrame, window_size: int) -> pd.Series:
    """波峰因子"""
    p = pd.Series(0, index=data_dic.index)
    vol_mean = ta.MA(data_dic['volumes'], window_size)
    vol_std = ta.STDDEV(data_dic['volumes'], window_size)
    p[data_dic['volumes'] > vol_mean + vol_std] = 1
    return ta.SUM(p, window_size)


def cor_vc(data_dic: pd.DataFrame, window_size) -> pd.Series:
    """量价相关因子"""
    minute_trade = ta.SUM(data_dic.total_volume_trade, 20)  # 分钟交易量
    dav = (data_dic['price'] - data_dic['price'].shift(window_size)) * minute_trade
    vol_std = ta.STDDEV(data_dic.total_volume_trade, 20)
    last_std = ta.STDDEV(data_dic['last'], 20)
    vc = ta.SUM(dav, window_size) / (vol_std * last_std)
    vc[np.isinf(vc)] = np.nan
    return vc


def disaster(minute_data: pd.DataFrame, window_size) -> pd.Series:
    """最优波动率"""
    results = []
    for i in range(window_size, len(minute_data)):
        window_data = minute_data.iloc[i - window_size:i][['price', 'last', 'high', 'low']].values.flatten()
        ratio_squared = (window_data.std() / window_data.mean()) ** 2
        results.append(ratio_squared)
    ratio = np.array([np.nan] * window_size + results)
    ratio = np.where(ratio == 0, np.nan, ratio)
    ratio = minute_data['r_minute'].values / ratio
    return ratio


def mpb(data_dic) -> pd.Series:
    """市价偏离度 Mid-Price Basis 中信建投-因子深度研究系列：高频量价选股因子初探"""
    ticks_num = 20
    va_t = ta.SUM(data_dic['values'], ticks_num)
    vol_t = ta.SUM(data_dic['volumes'], ticks_num)
    tp = va_t / vol_t  # 注意单位
    tp.fillna(method='ffill', inplace=True)
    tick_fac_data = tp - (data_dic['price'] + data_dic['price'].shift(ticks_num)) / 10000 / 2
    return tick_fac_data


def mpc(data_dic) -> pd.Series:
    """Midpoint Price Change 中信建投-高频选股因子分类体系"""
    ticks_num = 20
    b1 = data_dic['bid_price1']
    o1 = data_dic['offer_price1']
    if b1.isna().all() or o1.isna().all():
        tick_fac_data = 0
    else:
        mid = (ta.SMA(b1, ticks_num) + ta.SMA(o1, ticks_num)) / 2
        tick_fac_data = ta.ROCP(mid, ticks_num)
    return tick_fac_data


def mpc_max(data_dic: pd.DataFrame) -> pd.Series:
    b1 = data_dic['bid_price1']
    o1 = data_dic['offer_price1']
    if b1.isna().all() or o1.isna().all():
        mpcm = 0
    else:
        mpc_data = mpc(data_dic)
        if mpc_data.isna().all():
            mpcm = 0
        else:
            mpcm = ta.MAX(mpc_data, 20)
    return mpcm


def mpc_skew(data_dic: pd.DataFrame) -> pd.Series:
    b1 = data_dic['bid_price1']
    o1 = data_dic['offer_price1']
    if b1.isna().all() or o1.isna().all():
        mpcs = 0
    else:
        mpc_data = mpc(data_dic)
        if mpc_data.isna().all():
            mpcs = 0
        else:
            mpcs = cul_skew(mpc_data, 20)
    return mpcs


def mci_b(data_dic: pd.DataFrame) -> pd.Series:
    """Marginal Cost of Immediacy 边际交易成本 中信建投-高频选股因子分类体系"""
    mid = data_dic['price'] / 10000
    dol_vol_b = pd.Series(0, index=data_dic.index)
    q_bid = pd.Series(0, index=data_dic.index)
    for i in range(1, 6):
        dol_vol_b += data_dic['bid_price' + str(i)] * data_dic['bid_volume' + str(i)] / 10000
        q_bid += data_dic['bid_volume' + str(i)]
    v_wap = - (dol_vol_b / q_bid - mid) / mid * 100
    tick_fac = v_wap / dol_vol_b * 10000
    if tick_fac.isna().all():
        tick_fac = 0
    return tick_fac


def ptor(data_dic: pd.DataFrame, r_minute) -> pd.Series:
    ticks_num = 100
    amt = ta.SMA(data_dic['values'], ticks_num)
    tv = ta.SMA(data_dic['num_trades'], ticks_num)
    amt_per_trade = amt / tv
    amt_per_trade.fillna(0, inplace=True)
    apt_out = ta.SUM(data_dic['values'][r_minute < 0], ticks_num) / \
              ta.SUM(data_dic['num_trades'][r_minute < 0], ticks_num)
    tick_fac = apt_out / amt_per_trade
    tick_fac.fillna(0, inplace=True)
    return tick_fac


def bni(data_dic: pd.DataFrame, r_minute, window_size) -> pd.Series:
    """BNI 大资金流入比例"""
    ticks_num = 20
    amt = ta.SUM(data_dic['valeus'], ticks_num)
    tv = ta.SUM(data_dic['num_trades'], ticks_num)
    apt = amt / tv
    amt_big = pd.Series(0, index=data_dic.index)
    tv_big = pd.Series(0, index=data_dic.index)
    big_rank = apt.rolling(window=window_size).apply(lambda x: np.percentile(0.7))
    amt_big[(apt > big_rank) & (r_minute < 0)] = -data_dic['values'][
        (apt > big_rank) & (r_minute < 0)]
    amt_big[(apt > big_rank) & (r_minute > 0)] = data_dic['values'][
        (apt > big_rank) & (r_minute > 0)]
    tv_big[apt > big_rank] = data_dic['num_trades'][apt > big_rank]
    amt_big = ta.SUM(amt_big, ticks_num) / ta.SUM(tv_big, window_size)
    amt_big.fillna(0, inplace=True)
    tick_fac = amt_big / apt
    return tick_fac


def mb(data_dic: pd.DataFrame, window_size) -> pd.Series:
    """MB 大资金驱动涨幅"""
    ticks_num = 20
    r = np.log(data_dic['price'] / data_dic['price'].shift(1))
    amt = ta.SUM(data_dic['values'], ticks_num)
    tv = ta.SUM(data_dic['num_trades'], ticks_num)
    apt = amt / tv
    mbs = pd.Series(0, index=data_dic.index)
    for da, group in data_dic.groupby('date'):
        apt_big = apt[group.index].rank(ascending=False)
        big_rank = 0.3 * len(group)
        mbs[group[apt_big < big_rank].index] = r[group[apt_big < big_rank].index] / ticks_num
    mbs = ta.SUM(mbs, window_size)
    return mbs


def bam(data_dic: pd.DataFrame, window_size) -> pd.Series:
    buy_positive = pd.DataFrame(0, columns=['t1', 't2', 't3'], index=data_dic.index)
    tvt = ta.SUM(data_dic['total_value_trade'], window_size)
    trade1 = data_dic['total_value_trade'][data_dic['bid_price1'].shift(20) < data_dic['last']]
    trade2 = data_dic['total_value_trade'][data_dic['offer_price1'].shift(20) > data_dic['last']]
    trade3 = data_dic['total_value_trade'].drop(pd.concat([trade1, trade2]).drop_duplicates().index) / 2
    buy_positive.loc[trade1.index, 't1'] = trade1
    buy_positive.loc[trade2.index, 't2'] = trade2
    buy_positive.loc[trade3.index, 't3'] = trade3
    bam_t = ta.SUM(buy_positive['t1'] + buy_positive['t3'], window_size) / tvt
    # sam_t = ta.MA(buy_positive['t2']+buy_positive['t3'], window_size) / tvt
    return bam_t


def ba_cov(data_dic: pd.DataFrame, window_size) -> pd.Series:
    buy_positive = pd.DataFrame(0, columns=['t1', 't2', 't3'], index=data_dic.index)
    trade1 = data_dic['total_value_trade'][data_dic['bid_price1'].shift(20) < data_dic['last']]
    trade2 = data_dic['total_value_trade'][data_dic['offer_price1'].shift(20) > data_dic['last']]
    trade3 = data_dic['total_value_trade'].drop(pd.concat([trade1, trade2]).drop_duplicates().index) / 2
    buy_positive.loc[trade1.index, 't1'] = trade1
    buy_positive.loc[trade2.index, 't2'] = trade2
    buy_positive.loc[trade3.index, 't3'] = trade3
    ba = buy_positive['t1'] + buy_positive['t3']
    # sa = buy_positive['t2']+buy_positive['t3']
    ba = ta.MA(ba, window_size) / ta.STDDEV(ba, window_size)
    # sa = ta.MA(sa, window_size) / ta.STDDEV(sa, window_size)
    ba.fillna(method='ffill', inplace=True)
    return ba


def por(data_dic: pd.DataFrame, tick_nums) -> pd.Series:
    """积极买入成交额占总成交额的比例"""
    buy_positive = pd.Series(0, index=data_dic.index)
    op = data_dic['offer_price1']
    last = data_dic['last']
    tvt = data_dic['total_value_trade']
    buy_positive[last >= op.shift(1)] = tvt[last >= op.shift(1)]
    tick_fac_data = ta.SUM(buy_positive, tick_nums) / ta.SUM(tvt, tick_nums)
    return tick_fac_data


def returns_stock(data: pd.DataFrame, factor: str) -> pd.DataFrame:
    sto = []
    k = 10
    fac = data[['date', 'tick', 'r_pre', factor]].sort_values(factor, ascending=False)
    fac.reset_index(drop=True, inplace=True)
    for (da, ti), group in fac.groupby(['date', 'tick']):
        group_size = len(group) // k
        remainder = len(group) % k
        start_index = 0
        stocks = []
        for i in range(k):
            if i < remainder:
                end_index = start_index + group_size + 1
            else:
                end_index = start_index + group_size
            stocks += [group[start_index:end_index]['r_pre'].mean()]
            start_index = end_index
        sto.append(stocks + [da, ti])
    sto = pd.DataFrame(sto, columns=['r_pre' + str(s) for s in range(k)]+['date', 'tick'])
    sto = sto.groupby(['date', 'tick']).mean()
    sto = sto / 20
    return sto


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

        total_value_trade_ms = g['total_value_trade']
        groups['pearson'] = ta.CORREL(total_value_trade_ms, g.price, window_size)
        groups['voi'] = voi(g)
        groups['sori'] = sori(g)
        groups['mpc_skew'] = mpc_skew(g)
        groups['bam'] = bam(g, 20)
        groups['por'] = por(g, 20)

        lock.acquire()
        r_data.append(groups)
        lock.release()


def tick_handle(tick: pd.DataFrame, window_size):
    # 多线程计算，cores为线程数，一般为设置为cpu核心数，x86架构下可以提升运算速度
    glob_f = ['voi', 'sori', 'pearson', 'bam', 'mpc_skew', 'por']
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
    tick = pd.concat([tick, fac_neutral(tick, glob_f)], axis=1, copy=False)  # 中性化，多线程建议使用fac_neutral2
    return tick


def twap(tick: pd.DataFrame):
    price = tick['last'].mean()
    return price / 10000.0


def vwap(tick: pd.DataFrame):
    price = tick.iloc[-1]['total_value_trade'] / tick.iloc[-1]['total_volume_trade']
    return price


def dynamic_factor(tick: pd.DataFrame, trade: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8,
                   factor='sori_neutral_rank'):
    bid_price1 = tick['bid_price1'].values
    ask_price1 = tick['offer_price1'].values
    bid_vol1 = tick['bid_volume1'].values
    last = tick['last'].values
    time = pd.to_datetime(tick['time']/1000, format='%H%M%S').dt.time.values
    ret = (tick['price'] / tick['price'].shift(1))
    ret = ret.fillna(1).values
    factor_tick = tick[factor].values
    nums = len(tick)
    total_volume = tick.iloc[-1]['total_volume_trade'] * 0.2
    remain_volume = total_volume
    total_value = 0
    sr = 1
    t2 = 40
    withdraw = 0
    price_seq = np.array([])  # 挂单价格
    volume_seq = np.array([])  # 挂单量
    time_seq = np.array([])  # 挂单时间
    td_time = trade['time'].dt.time.values
    td_price = trade['trade_price'].values
    td_volume = trade['trade_volume'].values
    wd = []
    wdr = 0
    for i in range(nums):
        sr *= ret[i]  # 计算累计收益率
        if i >= lng:
            sr /= ret[i - lng]
        if np.isnan(factor_tick[i]):
            continue
        if len(price_seq) > 0:
            time_seq += 1
            td_index = np.where((td_time < time[i]) & (td_time >= time[i-1]) & (td_price >= min(price_seq)))
            if len(td_index[0]) > 0:
                trade_price = td_price[td_index]
                trade_volume = td_volume[td_index]
                s = 0
                j = 0
                k = 0
                sell_volume1 = np.where(price_seq <= max(trade_price), volume_seq, 0)
                while k < len(sell_volume1) and j < len(trade_volume):
                    if price_seq[k] <= trade_price[j]:
                        if s + volume_seq[k] <= trade_volume[j]:
                            s += volume_seq[k]
                            volume_seq[k] = 0
                            k += 1
                        else:
                            volume_seq[k] = s + volume_seq[k] - trade_volume[j]
                            sell_volume1[k] = trade_volume[j] - s
                            s = 0
                            j += 1
                    else:
                        j += 1
                        s = 0
                while k + 1 < len(sell_volume1) and sell_volume1[k + 1] > 0:
                    sell_volume1[k + 1] = 0
                    k += 1
                remain_volume -= np.sum(sell_volume1)
                total_value += sell_volume1 @ price_seq
            # 挂单量为0或挂单时间为20tick的撤单
            del_seq = np.where(volume_seq == 0)
            price_seq = np.delete(price_seq, del_seq)
            time_seq = np.delete(time_seq, del_seq)
            volume_seq = np.delete(volume_seq, del_seq)
            del_seq = np.where(time_seq > t2)
            if len(del_seq[0]) > 0:
                withdraw += volume_seq[del_seq[0]][0]
                price_seq = np.delete(price_seq, del_seq)
                time_seq = np.delete(time_seq, del_seq)
                volume_seq = np.delete(volume_seq, del_seq)
        if total_value > 0:
            wdr = withdraw / (total_volume - remain_volume)
            wd.append(wdr)
        if sr >= 1 and factor_tick[i] > beta1:  # 缓慢上涨按ask_price卖出
            sell_volume = remain_volume / (nums - i) * 4
            sell_volume = sell_volume // 100 * 100
            if wdr > 0.4:
                sv = min(sell_volume * wdr, bid_vol1[i])
                sv = sv // 100 * 100
                remain_volume -= sv
                total_value += sv * bid_price1[i]
                sell_volume = sell_volume - sv
            if nums - i > t2:
                if len(price_seq) > 0:  # 按顺序插入挂单
                    insert_index = np.searchsorted(price_seq, ask_price1[i] + 1)
                    price_seq = np.insert(price_seq, insert_index, ask_price1[i])
                    volume_seq = np.insert(volume_seq, insert_index, sell_volume)
                    time_seq = np.insert(time_seq, insert_index, 0)
                else:
                    price_seq = np.append(price_seq, ask_price1[i])  # 挂单
                    volume_seq = np.append(volume_seq, sell_volume)
                    time_seq = np.append(time_seq, 0)
        elif sr < 1 and factor_tick[i] > beta2:  # 快速下跌按照bid_price卖出
            p1 = total_value / (total_volume - remain_volume)
            p2 = np.mean(last[:i])
            bp = (p1 - p2) / p2 * 10000
            sell_volume = min(remain_volume / (nums - i) * 4, bid_vol1[i])
            sell_volume = sell_volume // 100 * 100
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price1[i]
        if remain_volume <= 0:
            break
    # if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
    #     total_value += remain_volume * tick['last'].iloc[-1]
    # price = total_value / total_volume
    price = total_value/(total_volume-remain_volume)
    withdraw = withdraw / (total_volume-remain_volume) * 100
    remain_volume = remain_volume / total_volume * 100
    return price / 10000.0, withdraw, remain_volume


def easy_test(tick: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8, factor='sori_neutral_rank'):
    bid_price1 = tick['bid_price1'].values
    bid_volume1 = tick['bid_volume1'].values
    ret = (tick['price'] / tick['price'].shift(1))
    ret = ret.fillna(1).values
    factor_tick = tick[factor].values
    nums = len(tick)
    total_volume = tick.iloc[-1]['total_volume_trade'] * 0.20
    remain_volume = total_volume
    total_value = 0
    sr = 1
    for i in range(nums):
        sr *= ret[i]  # 计算累计收益率
        if i >= lng:
            sr /= ret[i - lng]
        if np.isnan(factor_tick[i]):
            continue
        if (sr < 1 and factor_tick[i] > beta2) or (sr >= 1 and factor_tick[i] > beta1):  # 快速下跌按照bid_price卖出
            sell_volume = total_volume * (i / nums - 1) + remain_volume
            sell_volume = sell_volume // 100 * 100
            sell_volume = min(sell_volume, bid_volume1[i])
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price1[i]
        if remain_volume <= 0:
            break
    if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price / 10000.0


def easy_test2(tick: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8, factor='sori_neutral_rank'):
    last = tick['bid_price1'].values
    ret = (tick['price'] / tick['price'].shift(1))
    ret = ret.fillna(1).values
    factor_tick = tick[factor].values
    nums = len(tick)
    sr = 1
    price = []
    for i in range(nums):
        sr *= ret[i]  # 计算累计收益率
        if i >= lng:
            sr /= ret[i - lng]
        if np.isnan(factor_tick[i]):
            continue
        if (sr < 1 and factor_tick[i] > beta2) or (sr >= 1 and factor_tick[i] > beta1):  # 快速下跌按照bid_price卖出
            price.append(last[i])
    return np.mean(price) / 10000.0


def dynamic_twap(tick: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8, factor='sori_neutral_rank'):
    bid_price1 = tick['bid_price1'].values
    bid_price2 = tick['bid_price2'].values
    bid_price3 = tick['bid_price3'].values
    bid_price4 = tick['bid_price4'].values
    bid_price5 = tick['bid_price5'].values
    bid_vol1 = tick['bid_volume1'].values
    bid_vol2 = tick['bid_volume2'].values
    bid_vol3 = tick['bid_volume3'].values
    bid_vol4 = tick['bid_volume4'].values
    bid_vol5 = tick['bid_volume5'].values
    ask_price1 = tick['offer_price1'].values
    volume = tick['volumes'].values
    values = tick['values'].values
    tick_price = np.where(volume > 0, values / volume * 10000, 0)
    vol_ask = (tick_price - shift(bid_price1, 1, cval=0)) / (
            shift(ask_price1, 1, cval=0) - shift(bid_price1, 1, cval=0)) * volume
    vol_ask = np.where(vol_ask < 0, 0, vol_ask)
    vol_ask = np.where(tick_price > shift(ask_price1, 1, cval=0), volume, vol_ask)
    vol_ask = np.where(np.isnan(vol_ask), 0, vol_ask)
    ret = (tick['price'] / tick['price'].shift(1))
    ret = ret.fillna(1).values
    factor_tick = tick[factor].values
    nums = len(tick)
    total_volume = tick.iloc[-1]['total_volume_trade'] * 0.25
    remain_volume = total_volume
    total_value = 0
    sr = 1
    price_seq = np.array([])  # 挂单价格
    volume_seq = np.array([])  # 挂单量
    time_seq = np.array([])  # 挂单时间
    for i in range(nums):
        sr *= ret[i]  # 计算累计收益率
        if i >= lng:
            sr /= ret[i - lng]
        if np.isnan(factor_tick[i]):
            continue
        if len(volume_seq) > 0:  # 挂单量不为0
            time_seq += 1
            sell_volume1 = np.where(price_seq <= ask_price1[i-1], volume_seq, 0)
            s = 0  # 挂单成交量
            s5 = 0
            s4 = 0
            s3 = 0
            s2 = 0
            s1 = 0
            for k in range(len(sell_volume1)):  # 挂单序列循环
                if sum(sell_volume1) == 0:
                    break
                if price_seq[k] <= bid_price5[i-1]:
                    if s5 + sell_volume1[k] <= bid_vol5[i-1]:
                        s5 += sell_volume1[k]
                        volume_seq[k] = 0
                    else:
                        volume_seq[k] = s5 + sell_volume1[k] - bid_vol5[i-1]  # 剩余未卖出
                        sell_volume1[k] = bid_vol5[i-1] - s5  # 卖出份额
                        s5 += sell_volume1[k]
                elif bid_price5[i-1] < price_seq[k] <= bid_price4[i-1]:
                    if s4 + sell_volume1[k] <= bid_vol4[i-1]:
                        s4 += sell_volume1[k]
                        volume_seq[k] = 0
                    else:
                        volume_seq[k] = s4 + sell_volume1[k] - bid_vol4[i-1]  # 剩余未卖出
                        sell_volume1[k] = bid_vol4[i-1] - s4  # 卖出份额
                        s4 += sell_volume1[k]
                elif bid_price4[i-1] < price_seq[k] <= bid_price3[i-1]:
                    if s3 + sell_volume1[k] <= bid_vol3[i-1]:
                        s3 += sell_volume1[k]
                        volume_seq[k] = 0
                    else:
                        volume_seq[k] = s3 + sell_volume1[k] - bid_vol3[i-1]  # 剩余未卖出
                        sell_volume1[k] = bid_vol3[i-1] - s3  # 卖出份额
                        s3 += sell_volume1[k]
                elif bid_price3[i-1] < price_seq[k] <= bid_price2[i-1]:
                    if s2 + sell_volume1[k] <= bid_vol2[i-1]:
                        s2 += sell_volume1[k]
                        volume_seq[k] = 0
                    else:
                        volume_seq[k] = s2 + sell_volume1[k] - bid_vol2[i-1]  # 剩余未卖出
                        sell_volume1[k] = bid_vol2[i-1] - s2  # 卖出份额
                        s2 += sell_volume1[k]
                elif bid_price2[i-1] < price_seq[k] <= bid_price1[i-1]:
                    if s1 + sell_volume1[k] <= bid_vol1[i-1]:
                        s1 += sell_volume1[k]
                        volume_seq[k] = 0
                    else:
                        volume_seq[k] = s1 + sell_volume1[k] - bid_vol1[i-1]  # 剩余未卖出
                        sell_volume1[k] = bid_vol1[i-1] - s1  # 卖出份额
                        s1 += sell_volume1[k]
                else:
                    if s + sell_volume1[k] <= vol_ask[i]:  # vol_ask实际上估计的是上个ask_price1成交量
                        s += sell_volume1[k]
                        volume_seq[k] = 0
                    else:  # 超过volume的部分放到序列的第k个中
                        volume_seq[k] = s + sell_volume1[k] - vol_ask[i]  # 剩余未卖出
                        sell_volume1[k] = vol_ask[i] - s  # 卖出份额
                        s += sell_volume1[k]
                        break
            s = s + s1 + s2 + s3 + s4 + s5
            if k+1 < len(sell_volume1):  # 超过k的订单未卖出，卖出份额记为0
                for j in range(k+1, len(sell_volume1)):
                    sell_volume1[j] = 0
            remain_volume -= s
            total_value += sell_volume1 @ price_seq
            if time_seq[0] >= 20:
                price_seq = np.delete(price_seq, 0)
                volume_seq = np.delete(volume_seq, 0)
                time_seq = np.delete(time_seq, 0)
            # 挂单量未0的撤单
            price_seq = np.delete(price_seq, np.where(volume_seq == 0))
            time_seq = np.delete(time_seq, np.where(volume_seq == 0))
            volume_seq = np.delete(volume_seq, np.where(volume_seq == 0))
        if sr >= 1 and factor_tick[i] > beta1:  # 缓慢上涨按ask_price卖出
            sell_volume = remain_volume / (nums - i) * 2
            sell_volume = sell_volume // 100 * 100
            if i < nums-20:  # 最后时刻不挂单，只撤单
                if len(price_seq) > 0:
                    insert_index = np.searchsorted(price_seq, ask_price1[i]+1)
                    price_seq = np.insert(price_seq, insert_index, ask_price1[i])
                    volume_seq = np.insert(volume_seq, insert_index, sell_volume)
                    time_seq = np.insert(time_seq, insert_index, 0)
                else:
                    price_seq = np.append(price_seq, ask_price1[i])  # 挂单
                    volume_seq = np.append(volume_seq, sell_volume)
                    time_seq = np.append(time_seq, 0)
        elif sr < 1 and factor_tick[i] > beta2:  # 快速下跌按照bid_price卖出
            sell_volume = remain_volume / (nums - i) * 2
            if sell_volume > bid_vol1[i]:
                sell_volume = bid_vol1[i]
            sell_volume = sell_volume // 100 * 100
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price1[i]
        if remain_volume <= 0:
            break
    if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price / 10000.0
