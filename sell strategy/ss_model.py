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
    trade_data = pd.read_feather(trade_path + '/trade.feather')
    data = pd.read_feather(working_path + '/tickf.feather')

    fac = ['voi_neutral_rank', 'sori_neutral_rank', 'pearson_neutral_rank',
           'mpc_skew_neutral_rank', 'bam_neutral_rank', 'por_neutral_rank']
    data['port'] = data[fac].mean(axis=1)
    fac = fac + ['port']
    chg = []
    for (date, sec), group in data.groupby(['date', 'securityid']):
        print('日期：', date, '\t代码：', sec)
        tda = trade_data[(trade_data['date'] == date) & (trade_data['securityid'] == sec)]
        if np.isnan(group['bid_price1']).any() or np.isnan(group['offer_price1']).any():  # 跳过涨跌停的天
            continue
        price1 = twap(group)
        # price3 = easy_test(group)
        price2, withdraw, remain = dynamic_factor(group, tda, factor='port')
        bp = (price2 - price1) / price1 * 10000
        chg.append([sec, date, price1, price2, bp, withdraw, remain])
        # p = []
        # for name in fac:
        #     p.append((dynamic_twap(group, factor=name) - price1) / price1 * 10000)
        # chg.append([sec, date, price1] + p)
    # chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap']+fac)
    # print((chg[fac] > 0).sum() / len(chg))
    # print(chg[fac].mean())
    # chg.to_csv('model1.csv', index=False)
    chg = pd.DataFrame(chg, columns=['sec', 'date', 'twap', 'model', 'bp', 'withdraw', 'remain'])
    w = (chg['bp'] > 0).sum() / len(chg) * 100
    print("胜率：%.2f%%" % w)
    print("平均价格提高：%.2fbp" % chg['bp'].mean())
    print("剩余单量：%.2f%%" % chg['remain'].mean())
    print("撤单率：%.2f%%" % chg['withdraw'].mean())
