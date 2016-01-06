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

def deleteClass(X,y,num,c):
	"""Delete 'num' samples from class='class' in StudentLife dataset stress reports
	"""
	
	twoIndex=np.array([i for i in range(len(y)) if y[i]==c])
	np.random.shuffle(twoIndex)

	if num >= 0.7*len(twoIndex):
		print('Number of examples requested for delete too many. Exiting...')
		exit()

	delIndex=twoIndex[0:num]

	X=np.delete(X,delIndex,0)
	y=np.delete(y,delIndex,0)

	print(X.shape,y.shape)

	return(X,y)


def activeLabeling(y1,y2):
	y = np.abs(y1-y2)
	occurences = np.bincount(y)

	#Number of predictions in each class
	print(np.bincount(y1),np.bincount(y2))

	# Number of zeros and ones in occurences is the number of
	# examples they 'agreed' on in the Tolerance manner
	correct = occurences[0]+occurences[1]
	percent = float(correct)*100 / len(y)
	print(percent)



def main():
	print('-----------------------------')
	print('| Active Learning Activated |')
	print('-----------------------------')
	
	X = np.load('X.npy')
	Y = np.load('Y.npy')
	
	# fixes errors with Nan data
	X = preprocessing.Imputer().fit_transform(X)
	print(X.shape,Y.shape)

	#Deleting examples of majority class to enforce balance
	X,Y = deleteClass(X,Y,300,2)
	#X,Y = deleteClass(X,Y,40,1)


	
	# The feature division is not clear by their column number,
	# It was attempted intuitively while cross-checking with the 
	# feature_importance attribute to make two equally good subspaces 
	 
	# Features regarding the first classifier
	clasOneCols = [0,1,2,3,4,5,9,10,11,12,13,14,15,16,32]
	clasOneData= X[:,clasOneCols]

	# Features regarding the second classifier
	clasTwoCols = [6,7,8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
	clasTwoData= X[:,clasTwoCols]

	#print(clasOneData.shape, clasTwoData.shape)

	#assisning weights to penalize majority class over minority
	class_weights={0 : 1, 1 : 0.2 , 2 : 0.1 , 3 : 0.2, 4 :1}
	rfr1= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)
	rfr2= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)
	rfr3= RandomForestClassifier(n_estimators=300,class_weight=class_weights,n_jobs=-1)

	n_samples = 500
	rfr1.fit(clasOneData[1:n_samples],Y[1:n_samples])
	rfr2.fit(clasTwoData[1:n_samples],Y[1:n_samples])	
	rfr3.fit(X[1:n_samples],Y[1:n_samples])

	pred1 = rfr1.predict(clasOneData[n_samples:-1])
	print(tolAcc(Y[n_samples:-1],pred1))

	pred2 = rfr2.predict(clasTwoData[n_samples:-1])
	print(tolAcc(Y[n_samples:-1],pred2))

	pred3 = rfr3.predict(X[n_samples:-1])
	print(tolAcc(Y[n_samples:-1],pred3))

	pred1 = pred1.astype(np.int64)
	pred2 = pred2.astype(np.int64)
	activeLabeling(pred1,pred2)


if __name__ == '__main__':
	main()