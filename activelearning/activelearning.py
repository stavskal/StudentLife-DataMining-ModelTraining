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





def main():
	print('-----------------------------')
	print('| Active Learning Activated |')
	print('-----------------------------')
	
	X = np.load('X.npy')
	Y = np.load('Y.npy')
	
	# fixes errors with Nan data
	X = preprocessing.Imputer().fit_transform(X)
	print(X.shape,Y.shape)
	# Features regarding the first classifier
	clasOneCols = [0,1,2,3,4,5,9,10,11,12,13,14,15,16,32]
	clasOneData= X[:,clasOneCols]

	# Features regarding the second classifier
	clasTwoCols = [6,7,8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
	clasTwoData= X[:,clasTwoCols]

	print(clasOneData.shape, clasTwoData.shape)

	#assisning weights to penalize majority class over minority
	class_weights={0 : 1, 1 : 0.4 , 2 : 0.1 , 3 : 0.4, 4 :1}
	rfr1= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)
	rfr2= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)

	rfr1.fit(clasOneData[1:700],Y[1:700])
	rfr2.fit(clasTwoData[1:700],Y[1:700])

	pred1 = rfr1.predict(clasOneData[700:-1])
	print(tolAcc(Y[700:-1],pred1))

	pred2 = rfr2.predict(clasTwoData[700:-1])
	print(tolAcc(Y[700:-1],pred2))


if __name__ == '__main__':
	main()