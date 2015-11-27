import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from unbalanced_dataset import UnderSampler
import matplotlib.pyplot as plt
import time
import warnings

def tolAcc(y,pred):
	"""Returns accuracy as defined by the Tolerance Method
	   Input: ground truth, predition
	"""
	output = np.array(pred)
	scored = output - np.array(y)

	# Counting as correct predictions the ones which fall in +/-1, not only exact
	# I call it the 'Tolerance technique'
	correct=0
	c = Counter(scored)
	for k in c.keys():
		if k<2 and k>-2:
			correct += c[k]
	
	score = float(correct)/len(scored)
	return(score*100)

def fiPlot(rf):
	""" Bar plot of feature importances of RF
	"""
	fi = rf.feature_importances_
	print(len(fi))
	fi = 100* (fi/fi.max())
	sorted_idx = np.argsort(fi)
	pos = np.arange(len(fi))
	print(pos)
	plt.figure()
	plt.barh(pos,fi[sorted_idx],align='center')
	plt.savefig('featureImporances.png')



def main():
	X=np.load('numdata/epochFeats.npy')
	Y=np.load('numdata/epochLabels.npy')
	print(X.shape, Y.shape)
	labels= np.load('numdata/LOO.npy')
	folds=3

	#Pipeline stuff 
	forest = RandomForestClassifier(n_estimators=100, n_jobs = -1)
	scaler = preprocessing.StandardScaler()
	lolo = LeaveOneLabelOut(labels)
	kf = StratifiedKFold(Y,n_folds=folds)

	#us = UnderSampler(verbose=True)
#	pipe = Pipeline(steps=[('scaler',scaler),('pca',pca) ,('forest',forest)])
	bestNComp=[]
	for comp in range(2,25):
		pca= PCA(n_components=comp)
		pipe = Pipeline(steps=[('scaler',scaler),('pca',pca) ,('forest',forest)])

		score = cross_val_score(pipe,X,Y,cv=kf)
		bestNComp.append((score.mean(),comp))
	maxC= max([item[1] for item in bestNComp ])
	maxA= max([item[0] for item in bestNComp])
	print('Optimal number of Principal Components is: {0} with according accuracy of : {1}'.format(maxC,maxA))
#	X = preprocessing.scale(X)
	#X,Y = us.fit_transform(X,Y)

	"""
	

	for train_index,test_index in kf:
		Xtrain,Xtest = X[train_index], X[test_index]
		ytrain,ytest = Y[train_index], Y[test_index]
		forest.fit(Xtrain,ytrain)


		scores = forest.predict(Xtest)
		print(tolAcc(ytest,scores))
	"""





if __name__ == '__main__':
	main()