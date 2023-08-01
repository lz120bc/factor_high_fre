import os.path

from funs2 import *

if __name__ == "__main__":
    #声远机
    #working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
    #trade_path = '/Users/lvfreud/Desktop/中信建投/因子/data/trade'
    #trade_data = pd.read_feather(trade_path + '/trade.feather')
    #data = pd.read_feather(working_path + '/tickf.feather')

    #宇轩机
    working_path = 'D:\\中信建投实习\\bigdata\\tick'
    trade_path = 'D:\\中信建投实习\\bigdata\\trade'
    trade_data = pd.read_feather(trade_path + '\\trade.feather')
    data = pd.read_feather(working_path + '\\tickf.feather')

    fac = ['voi_neutral_rank', 'sori_neutral_rank',
           'mpc_skew_neutral_rank', 'bam_neutral_rank', 'por_neutral_rank']
    ric = [5.1, 9.2, 1.9, 2.5, 2.6]
    data['port'] = (data[fac] * ric).dropna().sum(axis=1) / np.sum(ric)
    chg = []
    for (date, sec), group in data.groupby(['date', 'securityid']):
        if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any():
            continue
        # tda = trade_data[(trade_data['date'] == date) & (trade_data['securityid'] == sec)]
        price1 = twap(group)
        price2, remain = easy_test(group, factor='port')
        # price2, withdraw, remain = dynamic_factor(group, tda, factor='port')
        bp = (price2 - price1) / price1 * 10000
        # print('日期：', date, '\t代码：', sec, end="\t")
        # print("%.2fbp\t剩余：%.2f%%\t撤单：%.2f%%" % (bp, remain, withdraw))
        chg.append([sec, date, price1, price2, bp, remain])
    chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap', 'model', 'bp', 'remain'])
    w = (chg['bp'] > 0).sum() / len(chg) * 100
    print("胜率：%.2f%%" % w)
    print("平均价格提高：%.2fbp" % chg['bp'].mean())
    print("剩余单量：%.2f%%" % chg['remain'].mean())
    # print("撤单率：%.2f%%" % chg['withdraw'].mean())
    # chg.to_csv('model1.csv', index=False)
