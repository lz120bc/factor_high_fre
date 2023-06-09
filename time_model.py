import pandas as pd
import numpy as np

working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
# data = pd.read_feather(working_path+'/tickf.feather')
# fac = ['pearson_neutral', 'voi_neutral', 'sori_neutral', 'bam_neutral', 'por_neutral', 'mpc_skew_neutral']
# dar = []
# for (date, time), group in data.groupby(['date', 'time']):
#     g = pd.DataFrame(index=group.index)
#     for factor in fac:
#         g[factor + '_rank'] = group[factor].rank(ascending=False) / len(g)
#     dar.append(g)
# dar = pd.concat(dar, axis=0)
# data = pd.concat([data, dar], axis=1)
# del dar
# data.to_feather(working_path+'/tickf.feather')
data = pd.read_feather(working_path+'/tickf.feather')
volume_tick = pd.read_feather(working_path+'/vwap.feather')
data['date'] = data['date'].astype('str')


def twap(tick: pd.DataFrame):
    price = tick['price'].mean()
    return price


def vwap(tick: pd.DataFrame):
    time = tick[['time']]
    time = time.merge(volume_tick, on='time', how='left')
    time_ratio = (time['vw'] / time['vw'].sum()).values
    price = (time_ratio * tick['price']).sum()
    return price


def dynamic_factor(tick: pd.DataFrame, alpha, beta=0.4, factor='sori_neutral_rank'):
    factor_tick = tick[factor].values
    price_tick = tick['price'].values
    nums = len(tick)
    total_volume = 10000000  # 10万手
    flag = (tick[factor] > beta).astype('int').values
    remain_volume = total_volume
    total_value = 0
    for i in range(nums):
        base_ratio = remain_volume / (nums - i)  # 根据时间均分
        adj_ratio = base_ratio * (1 + alpha * (factor_tick[i] - 0.5))
        sell_volume = adj_ratio * flag[i]
        sell_volume = sell_volume // 100 * 100
        if remain_volume < sell_volume or remain_volume == 0:
            total_value += remain_volume * price_tick[i]
            break
        remain_volume -= sell_volume
        total_value += sell_volume * price_tick[i]
    if remain_volume > 0:
        total_value += remain_volume * price_tick[nums-1]
    price = total_value / total_volume
    return price


def dynamic_vwap(tick: pd.DataFrame, alpha, beta=0.4, factor='sori_neutral_rank'):
    time = tick[['time']]
    time = time.merge(volume_tick, on='time', how='left')
    time_ratio = time['vw'] / time['vw'].sum()  # 标准化
    base = (time_ratio / (1 - time_ratio.cumsum())).values
    factor_tick = tick[factor].values
    price_tick = tick['price'].values
    nums = len(tick)
    total_volume = 10000000  # 10万手
    flag = (tick[factor] > beta).astype('int').values
    remain_volume = total_volume
    total_value = 0
    for i in range(nums):
        base_ratio = remain_volume * base[i]  # 根据时间均分
        adj_ratio = base_ratio * (1 + alpha * (factor_tick[i] - 0.5))
        sell_volume = adj_ratio * flag[i]
        sell_volume = sell_volume // 100 * 100
        if remain_volume < sell_volume or remain_volume == 0:
            total_value += remain_volume * price_tick[i]
            break
        remain_volume -= sell_volume
        total_value += sell_volume * price_tick[i]
    if remain_volume > 0:
        total_value += remain_volume * price_tick[nums-1]
    price = total_value / total_volume
    return price


chg = []
for (date, ss), group in data.groupby(['date', 'securityid']):
    if date == '2023-04-24':
        continue
    price1 = dynamic_vwap(group, 0.2, 0.6) / 10000.0
    price2 = vwap(group) / 10000.0
    bp = (price2 - price1) / price1 * 10000
    chg.append(bp)
chg = np.array(chg)
w = (chg > 0).sum() / len(chg) * 100
print("胜率：%.2f%%" % w)
print("平均价格提高：%.2fbp" % chg.mean())
