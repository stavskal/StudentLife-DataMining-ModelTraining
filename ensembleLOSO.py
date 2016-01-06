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

from pipeTrain import deleteClass


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
	X=np.load('numdata/withgps/epochFeats.npy')
	Y=np.load('numdata/withgps/epochLabels.npy')
	labels= np.load('numdata/withgps/LOO.npy')

	#fixes errors with Nan data
	X = preprocessing.Imputer().fit_transform(X)


	# Ensemble Random Forest Regressor stacked with Random Forest Classifier
	if sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		totalacc=0
	
		print(X.shape,Y.shape)

		# separating features into categories for Ensemble Training
		activityData = X[:,0:3 ]
		screenData = X[:,3:14]
		conversationData = X[:,14:20 ]
		colocationData = X[:,20:26]
		audioData = X[:,26:X.shape[1]]

		# I GOT IT FOR THE LOLO (baking soda)
		lolo = LeaveOneLabelOut(labels)	


		for traini, testi in lolo:
			np.random.shuffle(traini)
			# separating train data to 2 subsets: 
			# 1) Train RFR(60% of data)
			# 2) Get RF outputs with which train RF classifier (40% of data)
			train_index = traini[0: int(0.6*len(traini))]
			train_index2 =  traini[int(0.6*len(traini)):len(traini)]

			# Training 5 regressors on 5 types of features
			i=0
			for data in [activityData,screenData,conversationData,colocationData,audioData]:
				RF.append(RandomForestRegressor(n_estimators=30,n_jobs=-1))
				
				# Train RF regressor on first subset
				RF[i].fit(data[train_index],Y[train_index])
				# Get RF predictions on second subset
				outputRF.append( RF[i].predict(data[train_index2]) )
				# Get RF outputs on LOSO data
				outRFtest.append(RF[i].predict(data[testi]))
				i += 1

			middleTrainMat = np.transpose(np.array(outputRF))
			testMat = np.transpose(np.array(outRFtest))
		

			# RF classifier to combine regressors
			# assigning smaller weights to most popular class_weights	
			class_weights={0 : 1, 1 : 0.4 , 2 : 0.1 , 3 : 0.4, 4 :1}
			rfr= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)

			rfr.fit(middleTrainMat,Y[train_index2])
			pred = rfr.predict(testMat)

			# Print to screen mean error and Tolerance Score
			tempacc = tolAcc(Y[testi],pred,testMat)
			print(tempacc)
			totalacc += tempacc
			#print(rfr.feature_importances_)

			del outputRF[:]
			del outRFtest[:]
		print('LOSO TP accuracy: {0}'.format(totalacc/16))




if __name__ == '__main__':
	main(sys.argv[1:])