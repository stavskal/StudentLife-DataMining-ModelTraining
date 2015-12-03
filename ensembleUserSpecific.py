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
	#	print('Pred,True: {0},{1} Data: {2}'.format(pred[i],y[i],testMat[i,:]))
		if errors[i]<=1:
			correct+=1
		if errors[i]==0:
			truecorrect+=1
	score = float(correct)/len(y)
	truescore = float(truecorrect)/len(y)
	meanEr = sum(errors)/len(errors)
	#print('Mean error: {0}'.format(meanEr))
	#print('Min max prediction: {0},{1}'.format(min(pred),max(pred)))
	print('Truescore: {0}'.format(truescore))
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
	Y=np.load('numdata/epochLabels.npy')
	labels= np.load('numdata/LOO.npy')

	#for i in range(0,Y.shape[0]):
	#	if Y[i]==5 or Y[i]==4:
	#		Y[i]==0
	#	elif Y[i]==1:
	#		Y[i]==1
	#	else:
	#		Y[i]==2


	# Ensemble Random Forest Regressor stacked with Random Forest Classifier
	if sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		totalacc=0
	
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

		# Custom Cross-Validation
		# Indexes is used to split the dataset in a 40/40/20 manner
		# NOTE: 30/30/40 seemed to produce very similar results
		indexes = np.array([i for i in range(X.shape[0])])
		np.random.shuffle(indexes)

		# I GOT IT FOR THE LOLO (baking soda)
		lolo = LeaveOneLabelOut(labels)	
		#print(lolo,len(lolo))
		

		for traini, testi in lolo:
			#print(len(testi))
			np.random.shuffle(traini)
			# separating train data to 2 subsets: 
			# 1) Train RF (50% of data)
			# 2) Get RF outputs with which train RF classifier (50% of data)
			# separating data to 3 subsets: 
		
			train_index = testi[0: int(0.3*len(testi))]
			train_index2 =  testi[int(0.3*len(testi)):int(0.6*len(testi))]
			test_index = testi[int(0.6*len(testi)):len(testi)]
			print(len(train_index),len(train_index2),len(test_index))
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
			rfr= RandomForestClassifier(n_estimators=300,n_jobs=-1)
			rfr.fit(middleTrainMat,Y[train_index2])
			#print(middleTrainMat.shape)

			
			pred = rfr.predict(testMat)
			# Print to screen mean error and Tolerance Score
			tempacc = tolAcc(Y[test_index],pred,testMat)
			print(tempacc)
			totalacc += tempacc
			del outputRF[:]
			del outRFtest[:]
		print('LOSO TP accuracy : {0}'.format(totalacc/16))

if __name__ == '__main__':
	main(sys.argv[1:])