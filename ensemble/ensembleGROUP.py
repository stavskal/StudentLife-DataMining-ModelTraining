import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
import warnings


sys.path.insert(0,'//home/tabrianos/Desktop/Thesis/Database/venv/ADASYN/ADASYN')

from adasyn import ADASYN

def tolAcc(y,pred):
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
	y = y.astype(np.int64)
	pred = pred.astype(np.int64)
	#Number of predictions in each class
	print('              Ground Truth ------------ Prediction')
	print(np.bincount(y),np.bincount(pred))
	print(truescore*100)
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
	#Change to parent directory to load data
	os.chdir(os.path.pardir)
	X=np.load('numdata/withgps/epochFeats.npy')
	Y=np.load('numdata/withgps/epochLabels.npy')
	labels= np.load('numdata/withgps/LOO.npy')

	#fixes errors with Nan data
	X = preprocessing.Imputer().fit_transform(X)

	#BABY IM WORTH It NANANANAN
	adsn = ADASYN(ratio=0.7)
	X,Y = adsn.fit_transform(X,Y)
	print(X.shape,Y.shape)

	for i in range(0,Y.shape[0]):
		if Y[i]==5 or Y[i]==4:
			Y[i]=0
		elif Y[i]==1:
			continue
		else:
			Y[i]=2


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

		skf = StratifiedKFold(Y,n_folds=3)
		kf = KFold(X.shape[0],shuffle=True)
		for traini, testi in kf:
			print(len(traini),len(testi))
			#np.random.shuffle(traini)
			# separating train data to 2 subsets: 
			# 1) Train RFR(60% of data)
			# 2) Get RF outputs with which train RF classifier (40% of data)
			#train_index = traini[0: int(0.6*len(traini))]
			#train_index2 =  traini[int(0.6*len(traini)):len(traini)]

			class_weights={0 : 1, 1 : 0.4 , 2 : 0.1 , 3 : 0.4, 4 :1}
			rfr= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1)

			rfr.fit(X[traini],Y[traini])
			pred = rfr.predict(X[testi])

			# Print to screen mean error and Tolerance Score
			tempacc = tolAcc(Y[testi],pred)
			print(tempacc)
			totalacc += tempacc
			#print(rfr.feature_importances_)

		print('LOSO TP accuracy: {0}'.format(totalacc/3))




if __name__ == '__main__':
	main(sys.argv[1:])