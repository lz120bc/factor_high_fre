import datetime
import os.path

from funs2 import *

if __name__ == "__main__":
    working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick1'
    data = pd.read_feather(os.path.join(working_path, 'tickf.feather'))
    chg = []
    for (date, sec), group in data.groupby(['date', 'securityid']):
        if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any():
            continue
        if date < 20230601:
            price1 = twap(group)
            price2, remain, ratio = easy_test(group, factor='port')
            bp = (price2 - price1) / price1 * 10000
            chg.append([sec, date, ratio])
    chg = pd.DataFrame(chg, columns=['sec', 'date', 'ratio'])
    chg = chg.groupby('sec')['ratio'].mean().reset_index()
    chg.to_csv('model1.csv', index=False)
