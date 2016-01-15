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
	adsn = ADASYN(imb_threshold=0.5,ratio=0.7)
	X,Y = adsn.fit_transform(X,Y)
	X,Y = adsn.fit_transform(X,Y)
	X,Y = deleteClass(X,Y,100,2)

	print(Counter(Y))


	# Ensemble Random Forest Regressor stacked with Random Forest Classifier
	if sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		totalacc=0
	
		print(X.shape,Y.shape)

		n_folds=3
		skf = StratifiedKFold(Y,n_folds=n_folds)
		kf = KFold(X.shape[0],n_folds=n_folds,shuffle=True)
		for traini, testi in skf:
			print(len(traini),len(testi))

			# Although data is oversampled, still a small imbalance is present
			rfr= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1)

			rfr.fit(X[traini],Y[traini])
			pred = rfr.predict(X[testi])

			# Print to screen mean error and Tolerance Score
			tempacc = tolAcc(Y[testi],pred)
			print(tempacc)
			totalacc += tempacc

		print('LOSO TP accuracy: {0}'.format(totalacc/n_folds))




if __name__ == '__main__':
	main(sys.argv[1:])