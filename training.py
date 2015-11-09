import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import time



day = 86400
halfday = 43200
quarterday = 21600

times =[2*day]

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

uids1=['u59','u00','u08','u57','u52','u51','u36']

ch = [120,100,70,50,35]



# returns feature vector corresponing to (timestamp,stress_level) (report)
# This feature vector is of size mc(=Most Common), which varies due to Cross Validation.
# each cell corresponds to the % of usage for each app. Apps that were not used during 
# previous day have zero in feature vector cell
def appStatsL(cur,uid,timestamp,timeWin,mc):
	appOccurTotal = countAppOccur(cur,uid,timestamp,timeWin)
	keys = np.fromiter(iter(appOccurTotal.keys()), dtype=int)
	keys = np.sort(keys)
	appStats1 = np.zeros(len(keys))

	
	tStart = timestamp - timeWin

	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp <= %s AND time_stamp >= %s; """, [uid,timestamp,timestamp-day] )
	records= Counter( cur.fetchall() )

	for k in records.keys():
		records[k[0]] = records.pop(k)


	return records




#---------------------------------------------------------------------
# computes the total time (sec) that screen was on and off and the times it went on
def timeScreenLock(cur,uid,timestamp):
	#table name is in form: uXXdark
	uDark = uid +'dark'
	#tStart is meanStress(average report frequency) seconds before given timestamps
	tStart = timestamp - meanStress(cur,uid)

	#fetching all records that fall within this time period
	cur.execute('SELECT timeStart,timeStop FROM {0} WHERE timeStart >= {1} AND timeStop <= {2}'.format(uDark,tStart,timestamp))
	records = cur.fetchall()

	timesScreen = len(records)

	totalTime =0
	# each tuple contains the time period screen was on. Calculate its duration and add it to total
	for k in records:
		totalScreen += k[1]-k[0]

	uLock = uid + 'lock'
	#fetching all records that fall within this time period
	cur.execute('SELECT timeStart,timeStop FROM {0} WHERE timeStart >= {1} AND timeStop <= {2}'.format(uLock,tStart,timestamp))
	records = cur.fetchall()

	totalLock = 0
	# each tuple contains the time period screen was on. Calculate its duration and add it to total
	for k in records:
		totalLock += k[1]-k[0]

	timesLock = len(records)

	return(totalScreen,timesScreen,totalLock,timesLock)










#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()



# ------------TEST CASE-----------------------------
# A few users were picked from the dataset
# 70% of their stress reports and the corresponding features are used for training
# the rest 30% is used for testing. The train/test reports are randomly distributed
# throughout the whole experiment duration. No FV is used both for training and testing.
# After the 10 models are trained and tested, the overall accuracy is averaged
# Random Forests were picked due to their 'universal applicability', each with 25 decision trees


#TODO: maybe stick to a fixed number of apps and add more features such as screen on/off time(s), no of unique apps etc
# DO NOT FORGET  ----> IF ABOVE FEATURE VECTOR IS CONSTRUCTED TO MAKE IT ZERO MEAN

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#TODO: fix the goddamn k-fold or any proper train/test subsetting
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
accuracies = []
for mc in ch:

	acc=0
	totalP=0
	totalR=0
	maxminAcc =[]
	Xlist=[]
	for testUser in uids1:

		cur.execute("SELECT time_stamp,stress_level FROM {0}".format(testUser))
		records = cur.fetchall()
		meanTime = meanStress(cur,testUser)

		# Xtrain's rows are those FVs for ALL stress report timestamps 
		a=appStatsL(cur,testUser,records[0][0],meanTime,mc)

		
		#X,Y store initially the dataset and the labels accordingly
		Y = np.empty(len(records))
		X = np.array(records)

		# X is shuffled twice to ensure that the report sequence is close to random
		np.random.shuffle(X)
		np.random.shuffle(X)

		# Xlist contains Feature Vectors of many lengths according to each period
		t0 = time.time()
		for i in range(0,len(records)):
			Xlist.append( appStatsL(cur,testUser,X[i][0],meanTime,mc) )
			Y[i] = X[i][1]
		t1 = time.time()
		print('FV time: {0}'.format(t1-t0))


		print('Size of Y: {0}'.format(Y.shape))
		# Transforming Feature Vectors of different length to Bag-of-Apps (fixed)
		# for training and testing, Xtt
		t0 = time.time()
		Xtt = constructBOA(Xlist)
		t1 = time.time()
		print('BOA time: {0}'.format(t1-t0))
		print('size of Xtt: {0}'.format(Xtt.shape))
		t0 = time.time()
		Xtt = selectBestFeatures(Xtt, Xtt.shape[1]/2)
		t1 = time.time()
		print('Feature Reduction time: {0}'.format(t1-t0))
		print('New size of Xtt: {0}'.format(Xtt.shape))
	

		#defining length of Training/Test Set
		trainLength= int(0.7 * Xtt.shape[0])
		testLength= int(0.3 * Xtt.shape[0])

		#instantiating the appropriate matrices
		#Xtrain = np.empty([trainLength, len(a)], dtype=float)
		#Ytrain = np.empty([trainLength],dtype=int)

			
		#Xtest = np.empty([testLength, len(a)], dtype=float)
		#Ytest = np.empty(testLength,dtype=int)


		Xtrain = Xtt[0:trainLength , :]
		Ytrain = Y[ 0:trainLength ]

		print(trainLength,Xtt.shape[0])
		Xtest = Xtt [ trainLength:Xtt.shape[0], : ]
		Ytest = Y[ trainLength:Xtt.shape[0] ]

		print(Xtrain.shape, Ytrain.shape)
		print(Xtest.shape, Ytest.shape)
		#initiating and training forest, n_jobs indicates threads, -1 means all available
		forest = RandomForestClassifier(n_estimators=35, n_jobs = -1)
		forest = forest.fit(Xtrain,Ytrain)


		
		output = forest.predict(Xtest) 
			
		# because accuracy is never good on its own, precision and recall are computed
		#metricP = precision_score(Ytest,output, average='macro')
		#metricR = recall_score(Ytest,output, average='macro')

		tempAcc = forest.score(Xtest,Ytest)
		print('Accuracy: {0} %'.format(tempAcc*100))

		#totalP += metricP
		#totalR +=metricR
		acc += tempAcc
		maxminAcc.append(tempAcc*100)
		del Xlist[:]
		#print('User: {0}  Accuracy: {1}'.format(testUser,tempAcc))
	print('Average accuracy: {0} %  most common: {1}'.format(float(acc)*100/len(uids1), mc))
	print('Max / Min accuracy: {0}%  / {1}% '.format(max(maxminAcc), min(maxminAcc)))
	#print('Average precision: {0} %'.format(float(totalP)*100/len(uids1)))
	#print('Average recall: {0} %'.format(float(totalR)*100/len(uids1)))
	accuracies.append(float(acc)*100/len(uids1))


#x = np.array([i for i in range(0,len(accuracies))])
#y = np.asarray(accuracies)
#xtic = ['One day', '3/4 day','Half day', 'Quarter of day']
#plt.xticks(x, xtic)
#plt.plot(x,y)
#plt.savefig('trainingTimes.png')
