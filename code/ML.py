import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(sep):
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
	df1 = df.loc[df['time'] < datetime(year=2015, month=sep, day=1), :]
	df2 = df.loc[(df['time'] >= datetime(year=2015, month=sep, day=1)) & (df['time'] < datetime(year=2015, month=sep+1, day=1)), :]
	new_user = list(df2.mem_id.unique())

	sequence = []
	y = []
	grouped = pd.DataFrame(df1.groupby(['mem_id'])['brand'].apply(list))
	grouped['mem_id'] = grouped.index
	for index, value in grouped.iterrows():
		sequence.append(value['brand'])
		if value['mem_id'] in new_user:
			y.append(1)
		else:
			y.append(0)

	return (sequence, y, grouped)

def svc_param_selection(X, y, nfolds):
	model_to_set = SVC(kernel='rbf')
	parameters = {
		"C": [0.1, 1, 10, 100],
		"kernel": ['rbf','poly','sigmoid'],
		"gamma": [0.001, 0.01, 0.1, 1]
	}
	model_tunning = GridSearchCV(model_to_set, param_grid=parameters, cv=nfolds)
	model_tunning.fit(X, y)
	print(model_tunning.best_score_)
	print(model_tunning.best_params_)

def rf_param_selection(X, y, nfolds):
	model_to_set = RandomForestClassifier()
	parameters = {
		"max_depth": [3, None],
		"max_features": [1, 3, 10],
		"min_samples_split": [2, 3, 10],
		"min_samples_leaf": [1, 3, 10],
		"criterion": ["gini", "entropy"]
	}
	model_tunning = GridSearchCV(model_to_set, param_grid=parameters)
	model_tunning.fit(X, y)
	print(model_tunning.best_score_)
	print(model_tunning.best_params_)

if __name__ == '__main__':
	X, y, df = load_data(6)
	mlb = MultiLabelBinarizer()
	mlb.fit(X)
	X = mlb.transform(X)
	X = X.astype(np.int8)
	y = np.array(y)
	X, y = shuffle(X, y, random_state=0)
	# X_selection = X[0:300,:]
	# y_selection = y[0:300]
	# svc_param_selection(X_selection, y_selection, 10
	svc_classifier = SVC(kernel='rbf', C=0.1, gamma=0.1, probability=True)
	svc_classifier.fit(X, y)
	# rf_param_selection(X_selection, y_selection, 10)
	rf_classifier = RandomForestClassifier(criterion='gini', max_depth=None, max_features=10, min_samples_leaf=1, min_samples_split=3)
	rf_classifier.fit(X, y)
	
	df['mem_id'] = df['mem_id'].astype(str)
	for s in ['online', 'offline']:
		fh = open('/Users/jacob/Desktop/Python/Guangxi Market/data/original_'+s+'_2.txt')
		rows = fh.readlines()[1:]
		fh.close()
		users = []
		for row in rows:
			users.append(row.split(',')[1])
		existing_users = df.mem_id.unique()

		svc_predict = []
		svc_prob = []
		rf_predict = []
		rf_prob = []
		for u in users:
			if u not in existing_users:
				X_test = np.zeros((1,X.shape[1]), dtype=np.int)
			else:
				X_test = df.loc[df['mem_id']==u, 'brand']
				X_test = mlb.transform(X_test)
			svc_predict.append(svc_classifier.predict(X_test)[0])
			rf_predict.append(rf_classifier.predict(X_test)[0])
			if u not in existing_users:
				svc_prob.append(1)
				rf_prob.append(1)
			else:
				svc_prob.append(max(svc_classifier.predict_proba(X_test)[0]))
				rf_prob.append(max(rf_classifier.predict_proba(X_test)[0]))
		df_svc = pd.DataFrame({'user':users, 'class':svc_predict, 'prob':svc_prob})
		df_rf = pd.DataFrame({'user':users, 'class':rf_predict, 'prob':rf_prob})
		df_svc.to_csv('/Users/jacob/Desktop/Python/Guangxi Market/results/'+s+'_'+'5_SVM.txt', index=False)
		df_rf.to_csv('/Users/jacob/Desktop/Python/Guangxi Market/results/'+s+'_'+'5_RF.txt', index=False)
