import json,csv,sys,os,psycopg2,random
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import warnings
from collections import Counter
import pandas as pd
import seaborn as sns
from adasyn import ADASYN


def visualize(ac1,ac2,ac3):
	print(ac1)
	print(ac2)
	print(ac3)

	nac1 = np.asarray(ac1)
	nac2 = np.asarray(ac2)
	nac3 = np.asarray(ac3)

	together = np.concatenate((nac1,nac2,nac3),axis=0)

	df.append(pd.DataFrame({
		'Class1': nac1,
		'Class2': nac2,
		'combined': nac3,
		'percent' : pd.Categorical(['25' for i in range(0,len(nac1))]),
		}))

	print(df)


	x=np.arange(3)
	ax = plt.subplot(111)
	ax.bar(x-0.2,ac1,width=0.2,color='b',align='center')
	ax.bar(x,ac2,width=0.2,color='g',align='center')
	ax.bar(x+0.2,ac3,width=0.2,color='r',align='center')


	plt.savegif('baractive.png')



def tolAcc(y,pred):
	"""Returns accuracy as defined by the Tolerance Method
	   Input: ground truth, prediction
	"""
	correct=0
	truecorrect=0
	errors=[]
	for i in range(0,len(y)):
		errors.append(np.absolute(y[i]-pred[i]))
		if errors[i]<=1:
			correct+=1
		if errors[i]==0:
			truecorrect+=1
	score = float(correct)/len(y)
	truescore = float(truecorrect)/len(y)
	meanEr = sum(errors)/len(errors)
	return(score*100)




def activeLabeling(y1,y2):
	y = np.abs(y1-y2)
	occurences = np.bincount(y)

	#Number of predictions in each class
	#print(np.bincount(y1),np.bincount(y2))

	# Number of zeros and ones in occurences is the number of
	# examples they 'agreed' on in the Tolerance manner
	correct = occurences[0]+occurences[1]
	percent = float(correct)*100 / len(y)
	print('They agree: %s %%' % percent)
	correct1 = occurences[0]
	percent1 = float(correct1)*100 / len(y)
	print('Or do they? %s' % percent1)
	return(percent)

def deleteClass(X, y, num, c):
    """Delete 'num' samples from class=c in StudentLife dataset stress reports
    """

    twoIndex = np.array([i for i in range(len(y)) if y[i] == c])
    np.random.shuffle(twoIndex)

    delIndex = twoIndex[0:num]

    X = np.delete(X, delIndex, 0)
    y = np.delete(y, delIndex, 0)

    return(X, y)

def main():
	print('-----------------------------')
	print('| Active Learning Activated |')
	print('-----------------------------')
	
	X = np.load('X.npy')
	Y = np.load('Y.npy')
	print(Counter(Y))
	# fixes errors with Nan data
	X = preprocessing.Imputer().fit_transform(X)
	print(X.shape,Y.shape)

	adsn = ADASYN(ratio=0.7)
	X,Y = adsn.fit_transform(X,Y)
	print(Counter(Y))

	X,Y = deleteClass(X,Y,100,2)
	print(Counter(Y))
	
	# The feature division is not clear by their column number,
	# It was attempted intuitively while cross-checking with the 
	# feature_importance attribute to make two equally good subspaces 
	 
	# Features regarding the first classifier
	clasOneCols = [0,1,2,3,4,5,9,10,11,12,13,14,15,16,32]
	clasOneData= X[:,clasOneCols]

	# Features regarding the second classifier
	clasTwoCols = [6,7,8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
	clasTwoData= X[:,clasTwoCols]

	#print(clasOneData.shape, clasTwoData.shape)

	#assisning weights to penalize majority class over minority
	#class_weights={0 : 1, 1 : 0.2 , 2 : 0.1 , 3 : 0.2, 4 :1}
	rfr1= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1)
	rfr2= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1)
	rfr3= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1)

	n_samples = 700
	tolac1 = []
	tolac2 = []
	tolac3 = []
	rate =[]

	ranges=['33','25','20']
	df =[]
	for i in [3,4,5,10]:
		skf = StratifiedKFold(Y,n_folds=i,shuffle=True)
		for test,train in skf:
			#print(len(train),len(test), float(len(train))/len(test))
			rfr1.fit(clasOneData[train],Y[train])
			rfr2.fit(clasTwoData[train],Y[train])	
			rfr3.fit(X[train],Y[train])
			
			pred1 = rfr1.predict(clasOneData[test])
			tolac1.append(tolAcc(Y[test],pred1))
			#print('Tolerance accuracy 1: %s' % tolAcc(Y[test],pred1))

			pred2 = rfr2.predict(clasTwoData[test])
			tolac2.append(tolAcc(Y[test],pred2))
		#	print('Tolerance accuracy 2: %s' % tolAcc(Y[test],pred2))

			pred3 = rfr3.predict(X[test])
			tolac3.append(tolAcc(Y[test],pred3))
			#print('Combined: %s' % tolAcc(Y[test],pred3))

			pred1 = pred1.astype(np.int64)
			pred2 = pred2.astype(np.int64)
			aggreement_rate = activeLabeling(pred1,pred2)
			rate.append(aggreement_rate)
		#print(rfr3.feature_importances_)
	print(rate[0:3])
	print('Mean is : %s' % np.mean(rate[0:3]))
	print(rate[3:6])
	print('Mean is : %s' % np.mean(rate[3:7]))
	print(rate[9:12])
	print('Mean is : %s' % np.mean(rate[7:12]))
	print(rate[12:16])
	print('Mean is : %s' % np.mean(rate[12:-1]))



"""
		nac1 = np.asarray(tolac1)
		nac2 = np.asarray(tolac2)
		nac3 = np.asarray(tolac3)

		#together = np.concatenate((nac1,nac2,nac3),axis=0)

		df.append(pd.DataFrame({
			'Accuracy': nac1,
			#'Class' : pd.Categorical([1 for i in range(0,len(nac1))])
			}))

		df.append(pd.DataFrame({
			'Accuracy': nac2,
			#'Class' : pd.Categorical([2 for i in range(0,len(nac2))])
			}))

		df.append(pd.DataFrame({
			'Accuracy': nac3,
			#'Class' : pd.Categorical(['combined' for i in range(0,len(nac2))])
			}))

		del tolac1[:]
		del tolac2[:]
		del tolac3[:]

	#print(df)
	df1 = pd.concat(df)
	print(df1)
	cat1 = [1 for j in range(0,len(nac1))]
	cat2 = [2 for j in range(0,len(nac2))]
	cat3 =['combined' for j in range(0,len(nac3))]
	catego = cat1 + cat2 + cat3
	print(catego)
	percent = ['33' for i in range(0,3*3)] + ['25' for i in range(0,3*4)] +['20' for i in range(0,3*5)] +['10' for i in range(0,3*10)]
	cats = [1,1,1,2,2,2,'combined','combined','combined',1,1,1,1,2,2,2,2,'combined','combined','combined','combined',1,1,1,1,1,2,2,2,2,2,'combined','combined','combined','combined','combined',1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,'combined','combined','combined','combined','combined','combined','combined','combined','combined','combined']
	#per = pd.Categorical.from_array(['33','33','33','25','25','25','25','20','20','20','20','20'])
	df1['Category'] = cats
	df1['percent'] = percent
	print(df1)

	sns.set_style('whitegrid')
	ax = sns.barplot(x='percent', y='Accuracy',hue='Category',data=df1)
	plt.legend(loc='upper right')
	fig = ax.get_figure()
	#fig.set_size_inches(15,9)
	fig.savefig('activelearn1.png')
"""




if __name__ == '__main__':
	main()