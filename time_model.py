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


def dynamic_twap(tick: pd.DataFrame, lng: int = 150, beta1: float = 0.3, beta2: float = 0.8, factor='sori_neutral_rank'):
    bid_price = tick['bid_price1'].values
    bid_vol = tick['bid_volume1'].values
    ask_price = tick['offer_price1'].values
    volume = tick['volumes'].values
    values = tick['values'].values
    last = tick['last'].values
    tick_price = values / volume
    vol_ask = (ask_price - tick_price) / (ask_price - bid_price) * volume  # 估计askprice1成交量
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
    for i in range(nums):
        sr *= ret[i]
        if i >= lng:
            sr /= ret[i - lng]
        if np.isnan(factor_tick[i]):
            continue
        if sum(volume_seq) > 0:
            sell_volume = np.where(price_seq <= last[i], volume_seq, 0)  # 价格符合的挂单价格的volume
            s = 0  # 挂单成交量
            for k in range(len(sell_volume)):
                if sell_volume[k] == 0:
                    continue
                if s + sell_volume[k] <= vol_ask[i]:  # 未超过volume估计值，则该挂单全部卖出
                    s += sell_volume[k]
                    volume_seq[k] = 0
                else:
                    volume_seq[k] = s + sell_volume[k] - vol_ask[i]  # 剩余未卖出
                    sell_volume[k] = vol_ask[i] - s  # 卖出份额
                    s += sell_volume[k]
                    break
            if k+1 < len(sell_volume):  # 此后订单全都未卖出
                for j in range(k+1, len(sell_volume)):
                    sell_volume[j] = 0
            remain_volume -= s
            total_value += sell_volume @ price_seq
        if len(price_seq) > 20:  # 撤单
            price_seq = np.delete(price_seq, 0)
            volume_seq = np.delete(volume_seq, 0)
        if sr > 1 and factor_tick[i] > beta1:  # 缓慢上涨按ask_price卖出
            sell_volume = remain_volume / (nums - i)
            sell_volume = sell_volume // 100 * 100
            price_seq = np.append(price_seq, ask_price[i])  # 挂单
            volume_seq = np.append(volume_seq, sell_volume)
        elif sr <= 1 and factor_tick[i] > beta2:  # 快速下跌按照bid_price卖出
            sell_volume = remain_volume / (nums - i)
            if sell_volume > bid_vol[i]:
                sell_volume = bid_vol[i]
            sell_volume = sell_volume // 100 * 100
            remain_volume -= sell_volume
            total_value += sell_volume * bid_price[i]
        if remain_volume == 0:
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
for (date, sec), group in data.groupby(['date', 'securityid']):
    if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any():  # 跳过涨跌停的天
        continue
    price2 = dynamic_twap(group)
    price1 = twap(group)
    bp = (price2 - price1) / price1 * 10000
    chg.append([sec, date, price1, price2, bp])
chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap', 'model1', 'bp'])
w = (chg['bp'] > 0).sum() / len(chg) * 100
print("胜率：%.2f%%" % w)
print("平均价格提高：%.2fbp" % chg['bp'].mean())
chg.to_csv('model1.csv', index=False)
