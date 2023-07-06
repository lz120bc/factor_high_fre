import pandas as pd
working_path = 'D:\\中信建投实习\\中信实习-算法交易\\tick'
data = pd.read_feather(working_path+'\\tickda.feather')
tvt = data.groupby(['securityid', 'date'])['volumes'].sum().reset_index()
tvt = tvt.rename(columns={'volumes': 'volumes_day'})
data = data.merge(tvt, on=['securityid', 'date'], how='left')
data['vw'] = data['volumes'] / data['volumes_day']
vw = data.groupby(['tick'])['vw'].mean().reset_index()
vw.to_feather(working_path+"\\vwap.feather")
# calculate
