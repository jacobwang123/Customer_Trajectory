import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import MultiLabelBinarizer

X = []
y = []
with open('/Users/jacob/Desktop/Python/Guangxi Market/job/job_data_4.txt') as f:
	lines = f.readlines()
	for line in lines:
		line = line.rstrip('\n')
		l = line.split(',')
		l = list(map(int, l))
		X.append(l[:len(l)-1])
		y.append(l[-1])

X = MultiLabelBinarizer().fit_transform(X)
X = X.astype(np.int8)
df_X = pd.DataFrame(X)
del X
gc.collect()

y = np.array(y)
df_X['map'] = pd.Series(y)

header = """@RELATION market

"""
attr = ''
for i in range(df_X.shape[1] - 1):
	attr += '@ATTRIBUTE\t' + str(i) + '\tNUMERIC\n'
header += attr
attr = '@ATTRIBUTE\tclass\t{'
#class_list = list(df_X['brand'])
#class_list = list(set(class_list))
class_list = np.unique(y)
for i in class_list:
#	i = i.replace(' ', '_')
	attr += str(i) + ','
attr = attr.rstrip(',') + '}\n\n'
header += attr
header += '@DATA\n'
with open('/Users/jacob/Downloads/job_data_5.txt', 'w') as f:
	f.write(header)

df_X.to_csv('/Users/jacob/Downloads/job_data_5.txt', header=False, index=False, mode='a')

# with open('/Users/jacob/Downloads/job_data_5.txt', 'a') as f:
# 	for (i, j), value in np.ndenumerate(X):
# 		line = list(X[i,:])
# 		for element in line:
# 			f.write(str(element) + ',')
# 		f.write(str(y[i]) + '\n')
