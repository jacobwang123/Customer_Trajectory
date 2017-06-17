import numpy as np
import pandas as pd
from datetime import datetime

def get_top5_list(s):
	r_p = s.split(';')
	top5 = []
	for i in r_p:
		i = i.lstrip('(').rstrip(')')
		top5.append(i.split(',')[0])
	return top5
	
df1 = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data1.txt', sep=',', header=0)
df2 = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data2.txt', sep=',', header=0)
df = pd.concat([df1, df2])
map_df = df[['brand','map']].drop_duplicates()

df_test = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/results/test_results.txt',
					sep=':', header=None, names=['mem_id','top5'])
df_test['top5'] = df_test['top5'].map(get_top5_list)

del df2['time']
del df2['map']
df2 = pd.DataFrame(df2.groupby(['mem_id'])['brand'].apply(list))
df2['mem_id'] = df2.index

df_match = pd.merge(df_test, df2)
qual_user = []
for index, row in df_match.iterrows():
	temp = list(set(row['top5']) & set(row['brand']))
	if len(temp) > 0:
		qual_user.append(row['mem_id'])