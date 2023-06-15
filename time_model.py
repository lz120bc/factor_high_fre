import pandas as pd
import numpy as np
import talib as ta

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


def dynamic_twap(tick: pd.DataFrame, beta: float, factor='sori_neutral_rank'):
    bid_price = tick['bid_price1'].values
    factor_tick = tick[factor].values
    nums = len(tick)
    total_volume = 10000000  # 10万手
    remain_volume = total_volume
    lng = 200
    total_value = 0
    sr = 0
    for i in range(nums):
        sr += factor_tick[i]
        if i >= lng:
            sr -= factor_tick[i - lng]
        if np.isnan(bid_price[i]) or bid_price[i] == 0:  # 跌停不能卖出
            continue
        sell_volume = remain_volume / (nums - i)
        sell_volume = sell_volume // 100 * 100
        if sr <= beta or factor_tick[i] > 0.95:
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price[i]
        if remain_volume == 0:
            break
    if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price / 10000.0


def dynamic_vwap(tick: pd.DataFrame, beta: float, factor='sori_neutral_rank'):
    time = tick[['tick']]
    time = time.merge(volume_tick, on='tick', how='left')
    time_ratio = time['vw'] / time['vw'].sum()  # 标准化
    bid_price = tick['bid_price1'].values
    nums = len(tick)
    base = (time_ratio / (1 - time_ratio.cumsum()).shift(1)).shift(-1).values  # 计算w
    base[nums - 1] = 1
    total_volume = 10000000  # 10万手
    flag = (tick[factor] > beta).astype('int').values
    remain_volume = total_volume
    total_value = 0
    for i in range(nums):
        if np.isnan(bid_price[i]) or flag[i] == 0:  # 跌停不能卖出
            continue
        sell_volume = remain_volume * base[i]
        sell_volume = sell_volume // 100 * 100
        remain_volume -= sell_volume
        total_value += sell_volume * bid_price[i]
        if remain_volume == 0:  # 剩余为0退
            break
    if remain_volume > 0:
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price / 10000.0


# 寻参优化
# sori -0.75
# sori_rank 0.93
# for b in np.linspace(0, 1, 5):
#     chg = []
#     for (date, ss), group in data.groupby(['date', 'securityid']):
#         if date == '2023-04-24':
#             continue
#         price2 = dynamic_twap(group, 0, b)
#         price1 = twap(group)
#         bp = (price2 - price1) / price1 * 10000
#         chg.append(bp)
#     chg = np.array(chg)
#     w = (chg > 0).sum() / len(chg) * 100
#     print("b=%.2f" % b)
#     print("胜率：%.2f%%" % w)
#     print("平均价格提高：%.2fbp" % chg.mean())

chg = []
for (date, sec), group in data.groupby(['date', 'securityid']):
    if date == '2023-04-25':  # 第一天缺少较多数据
        price2 = dynamic_twap(group, 100)
        price1 = twap(group)
        bp = (price2 - price1) / price1 * 10000
        chg.append([sec, date, price1, price2, bp])
chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap', 'model1', 'bp'])
w = (chg['bp'] > 0).sum() / len(chg) * 100
print("胜率：%.2f%%" % w)
print("平均价格提高：%.2fbp" % chg['bp'].mean())
# chg.to_csv('model1.csv', index=False)
