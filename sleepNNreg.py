import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
from processingFunctions import *
import matplotlib.pyplot as pyp
import time
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LogisticRegression ,LinearRegression
import theano
import theano.tensor as T
from nolearn.lasagne import NeuralNet, TrainSplit
from nolearn.lasagne.visualize import plot_loss
import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
import seaborn as sns
from sklearn import preprocessing, linear_model
from sklearn.cross_validation import cross_val_score, KFold
# -----------------------------------------------------------------------------------
# This script is intended to train a non-linear estimator for sleep time during nights
# Multi-Layer Perceptron will be used for the estimation (sklearn)
# -----------------------------------------------------------------------------------


def loadSleepLabels(cur,uid):
	uid = uid+'sleep'

	cur.execute('SELECT hour,time_stamp FROM {0}'.format(uid))
	records = cur.fetchall()
	#records = sorted(records,key=lambda x:x[1])
	return(np.array(records)) 


# returns duration (seconds) screen remained locked during previous evening and night
# used for Sleep Estimator as feature
def screenLockDur(cur,uid,timestamp):
	totalDur = 0
	uidL = uid + 'lock'
	cur.execute('SELECT * FROM {0} WHERE timeStart>={1} AND timeStop <= {2}'.format(uidL, timestamp-86400, timestamp) )
	records = cur.fetchall()
	
	#keeping only records during night epoch ( 21:00 < t < 10:00)
	tStart = [item[0] for item in records]
	timeEpochs = epochCalc(tStart)

	
	for i in range(0,len(tStart)):
		#if timeEpochs[i][0] =='night':
		totalDur += records[i][1] -records[i][0]

	return(totalDur)


# computes duration (seconds) user was stationary during night epoch ( 21:00 < t < 10:00)
# Feature for Sleep Estimator NN
def stationaryDur(cur,uid,timestamp):
	totalDur = 0
	uidS = uid +'act'
	cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(uidS, timestamp-86400, timestamp) )
	records = cur.fetchall()
	
	tStart = [item[0] for item in records]
	timeEpochs = epochCalc(tStart)

	for i in range(1,len(records)):
		# if two consecutive samples are the same and equal to zero (stationary) then calculate duration
		if records[i][1]==0:
			totalDur += 1

	return totalDur


# computes duration (seconds) user's phone was in a silent environment during night epoch ( 21:00 < t < 10:00)
# Feature for Sleep Estimator NN
def silenceDur(cur,uid,timestamp):
	totalDur = 0
	uidSil = uid+'audio'
	cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(uidSil, timestamp-86400, timestamp) )
	records = cur.fetchall()

	tStart = [item[0] for item in records]
	timeEpochs = epochCalc(tStart)

	for i in range(1,len(records)):
		# if two consecutive samples are the same and equal to zero, also in night then their duration
		# is added to the total silence duration
		if records[i][1]==0:
			totalDur += 1 

	return totalDur

#calculates total time phone stayed in dark during night epoch (db is (tstart,tstop))
# Feature for Sleep Estimator NN
def darknessDur(cur,uid,timestamp):
	totalDur = 0
	uidS = uid+'dark'
	#Getting data from database within day period
	cur.execute('SELECT * FROM {0} WHERE timeStart>={1} AND timeStop<={2}'.format(uidS, timestamp-86400, timestamp) )
	records = cur.fetchall()

	#timeEpochs holds tuples of timestamps and their according epochs
	tStart = [item[0] for item in records]
	tStop = [item[1] for item in records]
	timeEpochs = epochCalc(tStart)
	timeEpochs1 = epochCalc(tStop)


	for i in range(0,len(records)):
		if timeEpochs[i][0]=='night' or timeEpochs1[i][0]=='night':
			totalDur += records[i][1] - records[i][0]

	return np.absolute(totalDur)


# returns total charge time during nigth epoch
# Feature for Sleep Estimator NN
def chargeDur(cur,uid,timestamp):
	totalDur = 0
	uidC = uid+'charge'
	#Getting data from database within day period
	cur.execute('SELECT * FROM {0} WHERE start_timestamp>={1} AND end_timestamp<={2}'.format(uidC, timestamp-86400, timestamp) )
	records = cur.fetchall()

	#timeEpochs holds tuples of timestamps and their according epochs
	tStart = [item[0] for item in records]
	tStop = [item[1] for item in records]
	timeEpochs = epochCalc(tStart)
	timeEpochs1 = epochCalc(tStop)


	for i in range(0,len(records)):
		if timeEpochs[i][0]=='night' or timeEpochs1[i][0]=='night':
			totalDur += records[i][1] - records[i][0]

	return totalDur


def visualize(y,errorList,predictions):
	y1=list(y)
	pyp.figure()
	#y1,pr = zip(*sorted(zip(y1,pr)))
	y1,errorList = zip(*sorted(zip(y1,errorList)))

	y1,predictions = zip(*sorted(zip(y1,predictions)))
	xA = np.linspace(0,len(y1),len(y1))

	pyp.plot(xA,y1,'g--',label='Labels')	
	#pyp.plot(xA,errorList,'r')
	pyp.plot(xA,predictions,'b--',label='Estimation')
	pyp.plot(xA,errorList,'r--',label='Error')


	pyp.title('Sleep Duration Estimation with RandomForestRegressor')
	pyp.xlabel(' User Reports')
	pyp.ylabel('Hours slept (sorted)')
	pyp.legend(loc=2)




#Function to fit regression Random Forest
def regression(X,y):
	
	print(X.shape,y.shape)
	score = 0
	folds=3
	forest = rfr(n_estimators=10)
		
	# Ensuring label percentage balance when K-folding
	skf = KFold( X.shape[0], n_folds=folds)
	for train_index,test_index in skf:
		Xtrain,Xtest = X[train_index], X[test_index]
		ytrain,ytest = y[train_index], y[test_index]
		
		Xtrain = np.array(Xtrain,dtype='float64')
		Xtest = np.array(Xtest,dtype='float64')
		#Xtrain[np.isinf(Xtrain)] = 0
		forest.fit(Xtrain,ytrain)


		error=0
		errorList =[]
		predictions= []
		for i in range(0,Xtest.shape[0]):
			a= np.transpose(Xtest[i,:].reshape(Xtest[i,:].shape[0],1))
			
			pr = forest.predict(a)
			temp_err=np.absolute(pr-ytest[i])*60
			errorList.append(temp_err)	
			predictions.append(pr)
			error += temp_err

		print('Average error in minutes: {0}'.format(error/Xtest.shape[0]))
		print('Max/min/median error: {0} , {1} , {2}'.format(max(errorList),min(errorList),np.median(errorList)))
		del errorList[:]
		del predictions[:]




def regressNN(X,y):
	layers_all = [('input',InputLayer),
				   ('dense',DenseLayer),
				   	('output',DenseLayer)]

	net = NeuralNet(layers = layers_all,
 					 input_shape = (None,X.shape[1]),
					 dense_num_units=3,
					 dense_nonlinearity=None,
					 regression=True,
					 update_momentum=0.9,
					 update_learning_rate=0.001,
	 				 output_nonlinearity=None,
 					 output_num_units=1,
 					 max_epochs=150)

	print(X.shape,y.shape)
	#net.fit(X,y)
	folds=3
	skf = KFold( X.shape[0], n_folds=folds)
	for train_index,test_index in skf:
		Xtrain,Xtest = X[train_index], X[test_index]
		ytrain,ytest = y[train_index], y[test_index]
		
		Xtrain = np.array(Xtrain,dtype='float64')
		Xtest = np.array(Xtest,dtype='float64')
		#Xtrain[np.isinf(Xtrain)] = 0
		net.fit(Xtrain,ytrain)


		error=0
		errorList =[]
		predictions= []
		for i in range(0,Xtest.shape[0]):
			a= np.transpose(Xtest[i,:].reshape(Xtest[i,:].shape[0],1))
			
			pr = net.predict(a)
			temp_err=np.absolute(pr-ytest[i])*60
			errorList.append(temp_err)	
			predictions.append(pr)
			error += temp_err

		print('Average error in minutes: {0}'.format(error/Xtest.shape[0]))
		print('Max/min/median error: {0} , {1} , {2}'.format(max(errorList),min(errorList),np.median(errorList)))
		del errorList[:]
		del predictions[:]

	
def main(argv):
	#connecting to database with error handling
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()



	if sys.argv[1]=='-train':
	
		#X = np.empty((len(uids1),4),dtype='float32')
		X =[]
		y= []
		for trainUser in uids1:
			print(trainUser)
			sleepL = loadSleepLabels(cur,trainUser)
 			y += [item[0] for item in sleepL] 

			# computing five features to be used for regression of Sleep time, during night epoch:
			# 1) Total time phone stayed in dark environment (darkDur)
			# 2) Total time phone remained locked (sld)
			# 3) Total time audio classifier outputed silence (silDur)
			# 4) Total time activity classifier outputed stationary (statDur)
			# 5) Total charge time

			for i in range(0,len(sleepL)):
				# the following variables hold the 5 aforementioned features
				# which are appended to X list, later transformed to Xtrain matrix
				sld = screenLockDur(cur,trainUser,sleepL[i][1])				
				statDur = stationaryDur(cur,trainUser,sleepL[i][1])
				silDur = silenceDur(cur,trainUser,sleepL[i][1])
				#darkDur = darknessDur(cur,trainUser,sleepL[i][1])
				#chDur = chargeDur(cur,trainUser,sleepL[i][1])
			
				#print([sld,darkDur,silDur,statDur],sleepL[i][0])
				X.append( [sld,silDur,statDur])
		
		# In the following steps, Nan values are replaced with zeros and
		# features are normalized (zero mena, std 1)
		# Also skewed FVs are removed from Train Matrix
		Xtrain = np.nan_to_num(X)
		Xtrain1 = np.empty((Xtrain.shape[0],Xtrain.shape[1]),dtype='float32')
		deleteList = []
		for i in range(0,Xtrain.shape[1]):
			if np.std(Xtrain[:,i])<0.01:
				deleteList.append(i)


		#deleting all 'defective' training examples
		Xtrain1 = np.delete(Xtrain1,deleteList,0)
		y1=np.array(y)
		y1=np.delete(y1,deleteList)

		Xtrain2 = preprocessing.scale(Xtrain1)
	
		
		#regressNN(Xtrain2,y1)
		regression(Xtrain1,y1)
		regression(Xtrain2,y1)
		







if __name__ == '__main__':
	main(sys.argv[1:])