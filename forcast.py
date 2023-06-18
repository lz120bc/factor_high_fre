import pandas as pd
import numpy as np
import talib as ta
import datetime

working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
data = pd.read_feather(working_path + '/tickf.feather')
data = data.sort_values(by=['date', 'tick'])
factor = ['r_minute', 'r_5', 'r_mean5', 'const', 'dsmv', 'sori_neutral', 'voi_neutral']
returns = ['r_pre']


def forcast(rol, xt):
    # 根据因子值回归预测收益率
    da = rol[factor + returns].dropna()
    x = da[factor].values
    y = da[returns].values
    if len(da) == 0:
        y_pred = 0
    else:
        beta = np.linalg.inv(x.T @ x) @ x.T @ y
        y_pred = xt @ beta
    return y_pred


sp = []
roll = pd.DataFrame()
n = 20
for (date, tick), group in data.groupby(['date', 'tick']):
    if tick == datetime.time(9, 30):
        roll = group
        continue
    xt = group[factor].values
    y_pre = forcast(roll, xt)
    y_pre = pd.DataFrame(data=y_pre, index=group.index, columns=['y'])
    sp.append(y_pre)
    roll = pd.concat([roll, group], axis=0, ignore_index=False)
    if tick >= datetime.time(9, 31):
        roll = roll.iloc[-n * 200:]
sp = pd.concat(sp, axis=0, ignore_index=False)
data = data.merge(sp, left_index=True, right_index=True, how='left')
data = data.reset_index(drop=True)
data.to_feather(working_path + '/tickf.feather')
