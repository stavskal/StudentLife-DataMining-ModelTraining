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


def visualizeErrDist(y,pred):
	x =[]
	for i in pred:
		if i>=0 and i<=1:
			x.append()


def tolAcc(y,pred,testMat):
	"""Returns accuracy as defined by the Tolerance Method
	   Input: ground truth, prediction
	"""
	correct=0
	truecorrect=0
	errors=[]
	for i in range(0,len(y)):
		errors.append(np.absolute(y[i]-pred[i]))
		print('Pred,True: {0},{1} Data: {2}'.format(pred[i],y[i],testMat[i,:]))
		if errors[i]<=1:
			correct+=1
		if errors[i]==0:
			truecorrect+=1
	score = float(correct)/len(y)
	truescore = float(truecorrect)/len(y)
	meanEr = sum(errors)/len(errors)
	print('Mean error: {0}'.format(meanEr))
	print('Min max prediction: {0},{1}'.format(min(pred),max(pred)))
	return((score*100,truescore*100))

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
		Y = np.load('numdata/epochLabels.npy')
		print(X.shape, Y.shape, labels.shape)
		folds=10

		#Pipeline stuff 
		forest = RandomForestRegressor(n_estimators=100, n_jobs = -1)
		scaler = preprocessing.StandardScaler()

		lolo = LeaveOneLabelOut(labels)
		
		acc = 0

		us = UnderSampler(verbose=True)

		X,Y = us.fit_transform(X,Y)
		kf = KFold(Y.shape[0],n_folds=folds)
		for train_index,test_index in kf:
		#	print max(train_index),max(test_index)
			Xtrain,Xtest = X[train_index], X[test_index]
			ytrain,ytest = Y[train_index], Y[test_index]
			
			forest.fit(Xtrain,ytrain)


			scores = forest.predict(Xtest)
			acc += tolAcc(ytest,scores)
			
		print(acc/folds)



	# Ensemble Random Forest Regressor stacked with Neural Network	
	elif sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		folds = 20
		Y = np.load('numdata/epochLabels.npy')
		#	Y=Y+1
		us = UnderSampler(verbose=True)
		#cc = ClusterCentroids(verbose=True)
		#X,Y = us.fit_transform(X,Y)
		print(X.shape,Y.shape)

		# separating features into categories for Ensemble Training
		activityData = X[:,0:3 ]
		screenData = X[:,3:14]
		conversationData = X[:,14:20 ]
		colocationData = X[:,20:26]
		audioData = X[:,26:X.shape[1]]

		# Indexes is used to split the dataset in a 40/40/20 manner
		# NOTE: 30/30/40 seemed to produce very similar results
		indexes = np.array([i for i in range(X.shape[0])])
		np.random.shuffle(indexes)

		# separating data to 3 subsets: 
		# 1) Train RF
		# 2) Get RF outputs with which train NN
		# 3) Test NN output on the rest
		train_index = indexes[0: int(0.3*X.shape[0])]
		train_index2 =  indexes[int(0.3*X.shape[0]):int(0.6*X.shape[0])]
		test_index = indexes[int(0.6*X.shape[0]):X.shape[0]]

		# Training 4 regressors on 4 types of features
		i=0
		for data in [activityData,screenData,conversationData,colocationData,audioData]:
			RF.append(RandomForestRegressor(n_estimators=300,max_features=None,n_jobs=-1))
			RF[i].fit(data[train_index],Y[train_index])
			outputRF.append( RF[i].predict(data[train_index2]) )
			outRFtest.append(RF[i].predict(data[test_index]))
			i += 1

		middleTrainMat = np.transpose(np.array(outputRF))
		testMat = np.transpose(np.array(outRFtest))
	
		layers_all = [('input',InputLayer),
					  ('dense',DenseLayer),
					  ('output',DenseLayer)]


		# Just a linear combination of the inputs is enough
		net = NeuralNet(layers = layers_all,
	 					 input_shape = (None,5),
						 dense_num_units=12,
						 dense_nonlinearity=None,
						 regression=True,
						 update_momentum=0.9,
						 update_learning_rate=0.001,
		 				 output_nonlinearity=None,
	 					 output_num_units=1,
	 					 max_epochs=60,) # 60 epochs is enough, usually converges around 30-40


		#net.fit(middleTrainMat,Y[train_index2])
		rfr= RandomForestClassifier(n_estimators=300,n_jobs=-1)
		rfr.fit(middleTrainMat,Y[train_index2])
		print(middleTrainMat.shape)
		net.fit(middleTrainMat,Y[train_index2])
		# save figure of learning curve
		visualizeError(net)
		
		pred = rfr.predict(testMat)
		# Print to screen mean error and Tolerance Score
		print(tolAcc(Y[test_index],pred,testMat))
		





if __name__ == '__main__':
	main(sys.argv[1:])