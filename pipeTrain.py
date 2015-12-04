import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from unbalanced_dataset import UnderSampler, ClusterCentroids
import matplotlib.pyplot as plt
import time
import warnings
from nolearn.lasagne import NeuralNet, TrainSplit
from nolearn.lasagne.visualize import plot_loss
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import softmax,sigmoid,tanh,rectify

def visualizeError(net):
	train_loss = np.array([i["train_loss"] for i in net.train_history_])
	valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
	plt.plot(train_loss, linewidth=3, label="train")
	plt.plot(valid_loss, linewidth=3, label="valid")
	plt.grid()
	plt.legend()
	plt.title('Learning Curve for Neural Network')
	plt.xlabel("epoch")
	plt.ylabel("loss")
#pyplot.ylim(1e-3, 1e-2)
	#plt.yscale("log")
	plt.savefig('trainvalloss.png')


def deleteClass(X,y,num,c):
	"""Delete 'num' samples from class='class' in StudentLife dataset stress reports
	"""
	
	twoIndex=np.array([i for i in range(len(y)) if y[i]==c])
	np.random.shuffle(twoIndex)

	if num >= 0.7*len(twoIndex):
		print('Number of examples requested for delete too many...')
		exit()


	delIndex=twoIndex[0:num]

	X=np.delete(X,delIndex,0)
	y=np.delete(y,delIndex,0)

	print(X.shape,y.shape)

	return(X,y)










def tolAcc(y,pred,testMat):
	"""Returns accuracy as defined by the Tolerance Method
	   Input: ground truth, prediction
	"""
	correct=0
	truecorrect=0
	errors=[]
	distY = np.zeros(5)
	distP = np.zeros(5)
	for i in range(0,len(y)):
		errors.append(np.absolute(y[i]-pred[i]))
		#	print('Pred,True: {0},{1} Data: {2}'.format(pred[i],y[i],testMat[i,:]))
		distP[pred[i]] += 1
		distY[y[i]] += 1
		if errors[i]<=1:
			correct+=1
		if errors[i]==0:
			truecorrect+=1
	score = float(correct)/len(y)
	truescore = float(truecorrect)/len(y)
	meanEr = sum(errors)/len(errors)
	print('Mean error: {0}'.format(meanEr))
	print('Prediction distribution: {0}'.format(distP/float(sum(distP))))
	print('Label distribution:      {0}'.format(distY/float(sum(distY))))

	return(score*100,truescore*100)

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
	Y=np.load('numdata/epochLabels.npy')
	labels= np.load('numdata/LOO.npy')
	print(X.shape,Y.shape)
	X,Y = deleteClass(X,Y,330,2)
	X,Y = deleteClass(X,Y,70,1)



	if sys.argv[1]=='-first':
		print(X.shape, Y.shape, labels.shape)
		folds=10
		#Pipeline stuff 
		forest = RandomForestRegressor(n_estimators=100, n_jobs = -1)
		scaler = preprocessing.StandardScaler()

		lolo = LeaveOneLabelOut(labels)	
		print(lolo,len(lolo))
		acc = 0

		us = UnderSampler(verbose=True)

		#X,Y = us.fit_transform(X,Y)
		kf = KFold(Y.shape[0],n_folds=folds)
		for train_index,test_index in lolo:

			print(len(train_index),len(test_index))
			Xtrain,Xtest = X[train_index], X[test_index]
			ytrain,ytest = Y[train_index], Y[test_index]
			
			forest.fit(Xtrain,ytrain)


			scores = forest.predict(Xtest)
			#acc += tolAcc(ytest,scores)
			
		print(acc/folds)



	# Ensemble Random Forest Regressor stacked with Random Forest Classifier
	elif sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
	
		us = UnderSampler(verbose=True)
		cc = ClusterCentroids(verbose=True)
		#X,Y = cc.fit_transform(X,Y)
		print(X.shape,Y.shape)

		# separating features into categories for Ensemble Training
		activityData = X[:,0:3 ]
		screenData = X[:,3:14]	
		conversationData = X[:,14:20 ]
		colocationData = X[:,20:26]
		audioData = X[:,26:X.shape[1]]

		# Custom Nested Cross-Validation
		# Indexes is used to split the dataset in a 40/40/20 manner
		# NOTE: 30/30/40 seemed to produce very similar results
		indexes = np.array([i for i in range(X.shape[0])])
		np.random.shuffle(indexes)

		lolo = LeaveOneLabelOut(labels)	
	#	print(lolo,len(lolo))
		# separating data to 3 subsets: 
		# 1) Train RF
		# 2) Get RF outputs with which train NN
		# 3) Test NN output on the rest
		train_index = indexes[0: int(0.5*X.shape[0])]
		train_index2 =  indexes[int(0.5*X.shape[0]):int(0.8*X.shape[0])]
		test_index = indexes[int(0.8*X.shape[0]):X.shape[0]]
		print(len(train_index),len(train_index2),len(test_index	))
		# Training 5 regressors on 5 types of features
		i=0
		for data in [activityData,screenData,conversationData,colocationData,audioData]:
			RF.append(RandomForestRegressor(n_estimators=300,max_features=data.shape[1],n_jobs=-1))
			RF[i].fit(data[train_index],Y[train_index])
			outputRF.append( RF[i].predict(data[train_index2]) )
			outRFtest.append(RF[i].predict(data[test_index]))
			i += 1

		middleTrainMat = np.transpose(np.array(outputRF))
		testMat = np.transpose(np.array(outRFtest))
	

		# RF classifier to combine regressors
		class_weights={0 : 1, 1 : 0.5 , 2 : 0.1 , 3 : 0.6, 4 :1}
		print(class_weights)
		rfr= ExtraTreesClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)
		rfr.fit(middleTrainMat,Y[train_index2])
		print(middleTrainMat.shape)

		
		pred = rfr.predict(testMat)
		# Print to screen mean error and Tolerance Score
		print(tolAcc(Y[test_index],pred,testMat))
		





if __name__ == '__main__':
	main(sys.argv[1:])