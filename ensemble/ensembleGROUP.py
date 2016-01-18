import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_fscore_support
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
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
	#print('              Ground Truth ------------ Prediction')
	#print(np.bincount(y),np.bincount(pred))
	return(score*100,truescore*100)


def deleteClass(X, y, num, c):
    """Delete 'num' samples from class=c in StudentLife dataset stress reports
    """

    twoIndex = np.array([i for i in range(len(y)) if y[i] == c])
    np.random.shuffle(twoIndex)

    delIndex = twoIndex[0:num]

    X = np.delete(X, delIndex, 0)
    y = np.delete(y, delIndex, 0)

    return(X, y)





def main(argv):
	#Change to parent directory to load data
	#os.chdir(os.path.pardir)
	X=np.load('data/X51.npy')
	Y=np.load('data/y51.npy')
	labels= np.load('data/LOO.npy')

	#fixes errors with Nan data
	X= preprocessing.Imputer().fit_transform(X)

	# Recursive oversampling and undersampling
	#adsn = ADASYN(imb_threshold=0.5,ratio=0.7)
	#X,Y = adsn.fit_transform(X,Y)
	#X,Y = adsn.fit_transform(X,Y)
	#X,Y = deleteClass(X,Y,100,2)
	

	for i in range(0,Y.shape[0]):
		if Y[i]==0 or Y[i]==1:
			Y[i]==0
		elif Y[i]==2:
			Y[i]=1
		else:
			Y[i]=2

	adsn = ADASYN(imb_threshold=0.4,ratio=1)
	X,Y = adsn.fit_transform(X,Y)
	adsn = ADASYN(imb_threshold=0.41,ratio=0.7)
	X,Y = adsn.fit_transform(X,Y)
	print(Counter(Y))
	
	# Ensemble Random Forest Regressor stacked with Random Forest Classifier
	if sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		totalacc=0
		totalRF=0
		totalXGB=0

		feats =[0,1,2,3,4,5,6,7,13,16,22,23,24,25,26,27,29,30,31,32,33,35,38,39,40,41,44,46,47,50]
		#X = X[:,feats]
		print(X.shape,Y.shape)

		n_folds=3
		skf = StratifiedKFold(Y,n_folds=n_folds)
		kf = KFold(X.shape[0],n_folds=n_folds,shuffle=True)
		for traini, testi in kf:
			print(len(traini),len(testi))

			# Although data is oversampled, still a small imbalance is present
			rfr= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1,criterion='entropy',max_features=X.shape[1],min_samples_split=1)
			gbm = xgb.XGBClassifier(n_estimators=50,learning_rate=0.5,colsample_bytree=0.3).fit(X[traini],Y[traini])

			rfr.fit(X[traini],Y[traini])
			pred = rfr.predict(X[testi])
			pred1 = gbm.predict(X[testi])
			# Print to screen mean error and Tolerance Score
			tempacc,trueRF = tolAcc(Y[testi],pred)
			print('Random Forest: %s' % tempacc)

			tempacc1,trueXGB = tolAcc(Y[testi],pred1)
			print('XGBoost: %s' % tempacc1)
			totalXGB += trueXGB
			totalRF += trueRF
			totalacc += tempacc

		print('True RF: {0}'.format(totalRF/n_folds))
		print('True XGB: {0}'.format(totalXGB/n_folds))
		print('LOSO TP accuracy: {0}'.format(totalacc/n_folds))

	elif sys.argv[1]=='-cali':
		# These parameters have been computed with RandomizedSearchCV
		rf_c= RandomForestClassifier(n_estimators=300,bootstrap=False,class_weight='auto',n_jobs=-1,criterion='entropy',max_features=10,min_samples_split=1)
		gbm = xgb.XGBClassifier(n_estimators=300,learning_rate=0.2,colsample_bytree=0.5, objective='multi:softmax',max_depth=10,gamma=0.001)
		Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.3)
		
		#Using isotonic calibration with 3-fold cv to improve results
		cc = CalibratedClassifierCV(rf_c,method='isotonic',cv=3)
		cc.fit(Xtrain,ytrain)
		pred = cc.predict(Xtest)
		print(precision_recall_fscore_support(ytest,pred))
		tolac,trueacc =tolAcc(ytest,pred)
		print(tolac)

		cc = CalibratedClassifierCV(gbm,method='isotonic',cv=3)
		cc.fit(Xtrain,ytrain)
		pred = cc.predict(Xtest)
		print(precision_recall_fscore_support(ytest,pred))
		tolac,trueacc =tolAcc(ytest,pred)
		print(tolac)

		#Comparing to not-calibrated xgboost
		gbm.fit(Xtrain,ytrain)
		pred = gbm.predict(Xtest)
		print(precision_recall_fscore_support(ytest,pred))
		tolac,trueacc =tolAcc(ytest,pred)
		print(tolac)






if __name__ == '__main__':
	main(sys.argv[1:])