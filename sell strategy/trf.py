from funs2 import *


if __name__ == "__main__":
    # tick trade合成
    tick = tick_data_handle(working_path='/Users/lvfreud/Desktop/中信建投/因子/data/tick', date_calculate="20230601_20230609")
    trade = trade_data_handle(trade_path='/Users/lvfreud/Desktop/中信建投/因子/data/trade', date="20230601_20230609")
    # data = pd.read_feather('/Users/lvfreud/Desktop/中信建投/因子/data/tick/tickf.feather')
    # trade = pd.read_feather('/Users/lvfreud/Desktop/中信建投/因子/data/tick/tradef.feather')
    tick = tick.merge(trade, how='left', on=['securityid', 'date', 'tick'])

    # 因子及中性化计算
    ws = 5 * 20
    factors = ['ptor']
    tick = tick_handle(tick, ws, glob_f=factors)

    # 计算IC/RankIC值
    factor_ic = []
    factors_neutral = [i + '_neutral' for i in factors]
    ic = [i + '_ic' for i in factors_neutral]
    rank_ic = [i + '_rank_ic' for i in factors_neutral]
    for (da, ti), group in tick.groupby(['date', 'tick']):
        fac = group[['r_pre']+factors_neutral].corr().iloc[0, 1:].to_list()
        ric = group[['r_pre']+factors_neutral].rank().corr().iloc[0, 1:].to_list()
        factor_ic.append([da, ti] + fac + ric)
    data_IC = pd.DataFrame(factor_ic, columns=['date', 'tick'] + ic + rank_ic)
    data_IC = data_IC.sort_values(['date', 'tick'])
    print(data_IC[rank_ic].mean())
    del factor_ic

    # 计算ICIR/RankICIR值
    data_ir = data_IC[ic].mean() / data_IC[ic].std()
    data_rank_ir = data_IC[rank_ic].mean() / data_IC[rank_ic].std()
    print(data_rank_ir)

    # 分组回测
    for kk in factors_neutral:
        sto = returns_stock(tick, kk)
        print(kk)
        print(sto.sum() / len(tick['date'].drop_duplicates()))
        sto.cumsum().plot().set_xticks([])
