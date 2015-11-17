import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
from processingFunctions import *
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.linear_model import LogisticRegression as lr
import theano
import theano.tensor as T
from nolearn.lasagne import NeuralNet
import lasagne
# -----------------------------------------------------------------------------------
# This script is intended to train a non-linear estimator for sleep time during nights
# Multi-Layer Perceptron will be used for the estimation (sklearn)
# -----------------------------------------------------------------------------------


def loadSleepLabels(cur,uid):
	uid = uid+'sleep'

	cur.execute('SELECT hour,time_stamp FROM {0}'.format(uid))
	records = cur.fetchall()
	records = sorted(records,key=lambda x:x[1])
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
		if timeEpochs[i][0] =='night':
			totalDur += records[i][1] -records[i][0]

	return(totalDur)


# computes duration (seconds) user was stationary during night epoch ( 21:00 < t < 10:00)
# Feature for Sleep Estimator NN
def stationaryDur(cur,uid,timestamp):
	totalDur = 0
	uidS = uid +'act'
	cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(uidS, timestamp-86400, timestamp) )
	records = cur.fetchall()
	
	#tStart = [item[0] for item in records]
	#timeEpochs = epochCalc(tStart)

	for i in range(1,len(records)):
		# if two consecutive samples are the same and equal to zero (stationary) then calculate duration
		if records[i-1][1] == records[i][1] and records[i][1]==0:
		
			totalDur += records[i][0] - records[i-1][0] 

	return totalDur


# computes duration (seconds) user's phone was in a silent environment during night epoch ( 21:00 < t < 10:00)
# Feature for Sleep Estimator NN
def silenceDur(cur,uid,timestamp):
	totalDur = 0
	uidSil = uid+'audio'
	cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(uidSil, timestamp-86400, timestamp) )
	records = cur.fetchall()

	#tStart = [item[0] for item in records]
	#timeEpochs = epochCalc(tStart)

	for i in range(1,len(records)):
		#if two consecutive samples are the same and equal to zero, also in night then their duration
		# is added to the total silence duration
		if records[i-1][1] == records[i][1] and records[i][1]==0:
			totalDur += records[i][0] - records[i-1][0] 

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
	timeEpochs = epochCalc(tStart)

	for i in range(0,len(records)):

		if timeEpochs[i][0]=='night':
			totalDur += records[i][1] - records[i][0] 

	return totalDur


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
	timeEpochs = epochCalc(tStart)

	for i in range(0,len(records)):
		if timeEpochs[i][0]=='night':
			totalDur += records[i][1] - records[i][0]

	return totalDur


#Function to fit regression NN with one hidden layer
def regression(X,y):
	error=0
	errorList =[]
	print(X.shape,y.shape)
	forest = rfr(n_estimators=20)
	forest.fit(X,y)

	for i in range(0,X.shape[0]):
		a= np.transpose(X[i,:].reshape(X[i,:].shape[0],1))
		
		pr = forest.predict(a)
		errorList.append(np.absolute(pr-y[i])*60)	
		error += np.absolute(pr-y[i])*60
		print(pr,y[i])
	print('Average error in minutes: {0}'.format(error/X.shape[0]))
	print('Max/min/median error: {0} , {1} , {2}'.format(max(errorList),min(errorList),np.median(errorList)))



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
				darkDur = darknessDur(cur,trainUser,sleepL[i][1])
				chDur = chargeDur(cur,trainUser,sleepL[i][1])

				X.append( [sld,darkDur,statDur,silDur,chDur])
		
		# In the following steps, Nan values are replaced with zeros and
		# feature vectors are normalized (zero mena, std 1)
		# Also skewed FVs are removed from Train Matrix
		Xtrain = np.nan_to_num(X)
		Xtrain1 = np.empty((Xtrain.shape[0],Xtrain.shape[1]),dtype='float32')
		deleteList = []
		for i in range(0,Xtrain.shape[0]):
			if np.std(Xtrain[i,:])>0:
				Xtrain1[i,:] = (Xtrain[i,:]-np.mean(Xtrain[i,:]))/np.std(Xtrain[i,:])
			else:
				deleteList.append(i)



		#deleting all 'defective' training examples
		Xtrain1 = np.delete(Xtrain1,deleteList,0)
		y1=np.array(y)
		y1=np.delete(y1,deleteList)
	
		

		regression(Xtrain1,y1)
		







if __name__ == '__main__':
	main(sys.argv[1:])