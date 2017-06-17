#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 12:12:42 2017

@author: jacob
"""

import numpy as np
import pandas as pd
from datetime import datetime

col_name = ['id', 'phone', 'mem_id', 'unknown', 'address', 'card', 'gender', 'age',\
			'brand', 'product', 'amount', 'price', 'store', 'time', 'date']
df = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/data/data.txt', sep=',',\
				 names=col_name, usecols=['mem_id', 'brand', 'time'])

df['time'] = pd.to_datetime(df['time'])

df['brand'] = df['brand'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
df['map'] = df[cat_columns].apply(lambda x: x.cat.codes)
print(df['brand'].nunique())

df = df.sort(['mem_id','time'], ascending=True)
boolean = df['time'] < datetime(year=2015, month=6, day=1)
df1 = df.loc[boolean, :]
df2 = df.loc[~boolean, :]
df1.to_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data1.txt', float_format='%i', index=False)
df2.to_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data2.txt', float_format='%i', index=False)

df = pd.read_csv('/Users/jacob/Desktop/Python/Guangxi Market/clean_data1.txt', sep=',', header=0)
df['time'] = pd.to_datetime(df['time'])
sequence = []
user = df.mem_id.unique()
for i in user:
	sub_df = df.loc[df['mem_id']==i, :]
	sub_df = sub_df.reset_index()
	timestamp = sub_df.time.unique()
	seq_num = len(timestamp)
	for j in range(0, seq_num):
		if j == seq_num-1:
			break
		else:
			move = list(sub_df.loc[sub_df['time']==timestamp[j], ['map']]['map'])
			next = list(sub_df.loc[sub_df['time']==timestamp[j+1], ['map']]['map'])
			for i in next:
				seq = move + [i]
				sequence.append(seq)

with open('/Users/jacob/Desktop/Python/Guangxi Market/job_data.txt', 'w') as f:
	for s in sequence:
		line = ''
		for i in s:
			line = line + str(i) + ','
		line = line.rstrip(',')
		f.write(line + '\n')

