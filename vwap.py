import pandas as pd
working_path = '/Users/lvfreud/Desktop/中信建投/因子/data/tick'
data = pd.read_feather(working_path+'/tickda.feather')
tvt = data.groupby(['securityid', 'date'])['volumes'].sum().reset_index()
tvt = tvt.rename(columns={'volumes': 'volumes_day'})
data = data.merge(tvt, on=['securityid', 'date'], how='left')
data['vw'] = data['volumes'] / data['volumes_day']
vw = data.groupby(['time'])['vw'].mean().reset_index()
vw.to_feather(working_path+"/vwap.feather")
