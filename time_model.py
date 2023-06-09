import pandas as pd
import numpy as np

working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
data = pd.read_feather(working_path+'/tickf.feather')  # 存放因子数据
volume_tick = pd.read_feather(working_path+'/vwap.feather')  # 根据历史数据构造
volume_tick.fillna(0, inplace=True)
data['date'] = data['date'].astype('str')


def twap(tick: pd.DataFrame):
    price = tick['last'].mean()
    return price


def vwap(tick: pd.DataFrame):
    price = tick.iloc[-1]['total_value_trade'] / tick.iloc[-1]['total_volume_trade']
    return price


def dynamic_twap(tick: pd.DataFrame, alpha: float, beta: float, factor='sori_neutral_rank'):
    factor_tick = tick[factor].values
    price_tick = tick['bid_price1'].values
    nums = len(tick)
    total_volume = 10000000  # 10万手
    flag = (tick[factor] > beta).astype('int').values
    remain_volume = total_volume
    total_value = 0
    for i in range(nums):
        if np.isnan(price_tick[i]):  # 跌停不能卖出
            continue
        base_ratio = remain_volume / (nums - i)  # 根据剩余时间均分
        adj_ratio = base_ratio * (1 + alpha * (factor_tick[i] - 0.5))
        sell_volume = adj_ratio * flag[i]
        sell_volume = sell_volume // 100 * 100
        if remain_volume < sell_volume or remain_volume == 0:  # 剩余为0或卖出大于剩余则退出
            total_value += remain_volume * price_tick[i]
            break
        remain_volume -= sell_volume
        total_value += sell_volume * price_tick[i]
    if remain_volume > 0:  # 如果存在未卖出股票，按收盘价卖出
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price


def dynamic_vwap(tick: pd.DataFrame, alpha: float, beta: float, factor='sori_neutral_rank'):
    time = tick[['time']]
    time = time.merge(volume_tick, on='time', how='left')
    time_ratio = time['vw'] / time['vw'].sum()  # 标准化
    factor_tick = tick[factor].values
    price_tick = tick['bid_price1'].values
    nums = len(tick)
    base = (time_ratio / (1 - time_ratio.cumsum()).shift(1)).shift(-1).values  # 计算w
    base[nums - 1] = 1
    total_volume = 10000000  # 10万手
    flag = (tick[factor] > beta).astype('int').values
    remain_volume = total_volume
    total_value = 0
    for i in range(nums):
        if np.isnan(price_tick[i]):  # 跌停不能卖出
            continue
        base_ratio = remain_volume * base[i]
        adj_ratio = base_ratio * (1 + alpha * (factor_tick[i] - 0.5))
        sell_volume = adj_ratio * flag[i]
        sell_volume = sell_volume // 100 * 100
        if remain_volume < sell_volume or remain_volume == 0:
            total_value += remain_volume * price_tick[i]
            break
        remain_volume -= sell_volume
        total_value += sell_volume * price_tick[i]
    if remain_volume > 0:
        total_value += remain_volume * tick['last'].iloc[-1]
    price = total_value / total_volume
    return price


# 寻参优化
# for a in np.linspace(0.1, 2.1, 10):
#     for b in np.linspace(0.3, 0.8, 5):
#         chg = []
#         for (date, ss), group in data.groupby(['date', 'securityid']):
#             if date == '2023-04-24':
#                 continue
#             price2 = dynamic_twap(group, a, b) / 10000.0
#             price1 = twap(group) / 10000.0
#             bp = (price2 - price1) / price1 * 10000
#             chg.append(bp)
#         chg = np.array(chg)
#         w = (chg > 0).sum() / len(chg) * 100
#         print("a=%.1f,b=%.1f" % (a, b))
#         print("胜率：%.2f%%" % w)
#         print("平均价格提高：%.2fbp" % chg.mean())

chg = []
for (date, _), group in data.groupby(['date', 'securityid']):
    if date == '2023-04-24':  # 第一天缺少较多数据
        continue
    price2 = dynamic_twap(group, 0.1, 0.6) / 10000.0
    price1 = twap(group) / 10000.0
    bp = (price2 - price1) / price1 * 10000
    chg.append([price1, price2, bp])
chg = pd.DataFrame(chg, columns=['twap', 'model1', 'bp'])
w = (chg['bp'] > 0).sum() / len(chg) * 100
print("胜率：%.2f%%" % w)
print("平均价格提高：%.2fbp" % chg['bp'].mean())
chg.to_csv('model1.csv', index=False)
