import pandas as pd
import numpy as np

working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
data = pd.read_feather(working_path + '/tickf.feather')  # 存放因子数据
volume_tick = pd.read_feather(working_path + '/vwap.feather')  # 根据历史数据构造
volume_tick.fillna(0, inplace=True)
data['date'] = data['date'].astype('str')


def twap(tick: pd.DataFrame):
    price = tick['last'].mean()
    return price / 10000.0


def vwap(tick: pd.DataFrame):
    price = tick.iloc[-1]['total_value_trade'] / tick.iloc[-1]['total_volume_trade']
    return price


def dynamic_twap(tick: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8, factor='sori_neutral_rank'):
    bid_price1 = tick['bid_price1'].values
    bid_vol = tick['bid_volume1'].values
    ask_price = tick['offer_price1'].values
    volume = tick['volumes'].values
    values = tick['values'].values
    tick_price = values / volume * 10000
    vol_ask = (tick_price - bid_price1) / (ask_price - bid_price1) * volume  # 估计每个tick按askprice1的成交量
    vol_ask = np.where(np.isnan(vol_ask), 0, vol_ask)
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
        if sr >= 1 and factor_tick[i] > beta1:  # 缓慢上涨按ask_price卖出
            sell_volume = remain_volume / (nums - i)
            sell_volume = sell_volume // 100 * 100
            s = 0
            t2 = 21
            if i + t2 >= nums:
                t2 = nums - i
            for j in range(1, t2):
                if ask_price[i] < bid_price1[i+j]:
                    s += volume[i+j]
                elif ask_price[i+j] >= ask_price[i] >= bid_price1[i+j]:
                    s += vol_ask[i+j]
            if sell_volume > s:
                sell_volume = s
            remain_volume -= sell_volume
            total_value += sell_volume * ask_price[i]
        elif sr < 1 and factor_tick[i] > beta2:  # 快速下跌按照bid_price卖出
            sell_volume = remain_volume / (nums - i)
            if sell_volume > bid_vol[i]:
                sell_volume = bid_vol[i]
            sell_volume = sell_volume // 100 * 100
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price1[i]
        if remain_volume <= 0:
            break
    if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price / 10000.0


# 寻参优化
# for b in range(100, 400, 50):
# for b in np.linspace(100, 400, 6):
#     chg = []
#     for (date, ss), group in data.groupby(['date', 'securityid']):
#         if date == '2023-04-24':
#             continue
#         price2 = dynamic_twap(group, b)
#         price1 = twap(group)
#         bp = (price2 - price1) / price1 * 10000
#         chg.append(bp)
#     chg = np.array(chg)
#     w = (chg > 0).sum() / len(chg) * 100
#     print("b=%.2f" % b)
#     print("胜率：%.2f%%" % w)
#     print("平均价格提高：%.2fbp" % chg.mean())

chg = []
fac = ['voi_neutral_rank', 'sori_neutral_rank', 'pearson_neutral_rank', 'mpc_skew_neutral_rank', 'bam_neutral_rank',
       'por_neutral_rank']
data['port'] = data[fac].mean(axis=1)
for (date, sec), group in data.groupby(['date', 'securityid']):
    if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any() or date == '2023-04-24':  # 跳过涨跌停的天
        continue
    price1 = twap(group)
    price2 = dynamic_twap(group, 200, 0.3, 0.8, 'sori_neutral_rank')
    bp = (price2 - price1) / price1 * 10000
    chg.append([sec, date, price1, price2, bp])
    # p = []
    # for name in fac:
    #     p.append((dynamic_twap(group, 150, 0.3, 0.8, name) - price1) / price1 * 10000)
    # chg.append([sec, date, price1] + p)
# chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap']+fac)
# chg.to_csv('model1.csv', index=False)
chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap', 'model', 'bp'])
w = (chg['bp'] > 0).sum() / len(chg) * 100
print("胜率：%.2f%%" % w)
print("平均价格提高：%.2fbp" % chg['bp'].mean())
