import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
	for i in range(0,len(y)):
		print(y[i],pred[i])
	output = np.array(pred)
	scored = output - np.array(y)

	# Counting as correct predictions the ones which fall in +/-1, not only exact
	# I call it the 'Tolerance technique'
	correct=0
	c = Counter(scored)
	for k in c.keys():
		if k<1.5 and k>-1.5:
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



def main(argv):
	X=np.load('numdata/epochFeats.npy')
	Y=np.transpose(np.array([np.load('numdata/epochLabels.npy')]))
	labels= np.transpose(np.array([np.load('numdata/LOO.npy')]))

	if sys.argv[1]=='-first':
		
		print(X.shape, Y.shape, labels.shape)
		folds=3

		#Pipeline stuff 
		forest = RandomForestRegressor(n_estimators=100, n_jobs = -1)
		scaler = preprocessing.StandardScaler()

		lolo = LeaveOneLabelOut(labels)
		
		acc = 0

		us = UnderSampler(verbose=True)
		X,Y = us.fit_transform(X,Y)
		kf = StratifiedKFold(Y,n_folds=folds)
		for train_index,test_index in kf:
		#	print max(train_index),max(test_index)
			Xtrain,Xtest = X[train_index], X[test_index]
			ytrain,ytest = Y[train_index], Y[test_index]
			
			forest.fit(Xtrain,ytrain)


			scores = forest.predict(Xtest)
			acc += tolAcc(ytest,scores)
			
		print(acc/folds)
	
	elif sys.argv[1]=='-ensemble':
		RF =[]
		
		print(X.shape,Y.shape,labels.shape)
		# concatenating user labels to distinguish which rows correspond to which user
		alltogether = np.concatenate((X,Y,labels),axis=1)
		print(alltogether.shape)
		for testUser in uids1:
			# separating user-specific data----------------------if u(XX) in Labels keep it
			trainMat = np.array([item[:] for item in alltogether if item[-1]==testUser[-2:]])
			
			# splitting again to keep only feature dataset, user labels not needed any more
			trainMat = trainMat[: , 0:trainMat.shape[1]-1]

			# separating features into categories for Ensemble Training
			activityData = trainMat[:,0:6 ]
			screenData = trainMat[:,6:17]
			conversationData = trainMat[:,17:23 ]
			colocationData = trainMat[:,23:trainMat.shape[1]-1]
			Y = trainMat[:,trainMat.shape[1]-1:trainMat.shape[1]]
			i=0
			#Training 4 regressors
			for data in [activityData,screenData,conversationData,colocationData]:
				RF.append(RandomForestRegressor(n_estimators=300,max_features=None,n_jobs=-1))
				RF[i].fit(data,Y)
				i += 1
			
			layers_all = [('input',InputLayer),
				   ('dense',DenseLayer),
				   	('output',DenseLayer)]

			net = NeuralNet(layers = layers_all,
 					 input_shape = (None,len(RF)),
					 dense_num_units=5,
					 dense_nonlinearity=None,
					 regression=True,
					 update_momentum=0.9,
					 update_learning_rate=0.001,
	 				 output_nonlinearity=None,
 					 output_num_units=1,
 					 max_epochs=100)


			print(activityData.shape, screenData.shape, conversationData.shape, colocationData.shape, Y.shape	)


	

"""
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

"""



if __name__ == '__main__':
	main()