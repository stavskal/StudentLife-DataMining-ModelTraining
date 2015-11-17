import datetime,psycopg2
from collections import Counter
import numpy as np
from sortedcontainers import SortedDict
#---------------------------------------------
#This script contains a collection of functions
#that are useful in processing the data in
#StudentLife Dataset

# Each functions has its description and internal comments
# follow the code as well.
#---------------------------------------------


# time lengths expressed in Seconds
hour = 3600
day = 86400
halfday = 43200
weekSec = 604000	


# List of all users in dataset
uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

# List of 'good' users
uids1=['u00','u24','u08','u57','u52','u51','u36','u59']

uids2=['u00','u24']

#---------------------------------------------------------------------------------------
# converts unix timestamp to human readable date (e.g '1234567890' -> '2009 02 14  00 31 30')
def unixTimeConv(timestamps):
	splitTimes = np.empty((len(timestamps), 7),dtype='float32')
	i=0
	for time in timestamps:
		newTime = str(datetime.datetime.fromtimestamp(int(time)))
		yearDate,timeT = newTime.split(' ')
		year,month,day = yearDate.split('-')
		hour,minutes,sec = timeT.split(':')
	#	print(type(hour))
		splitTimes[i,:] = (year,month,day,hour,minutes,sec,time)
		i += 1

	return (splitTimes)

#----------------------------------------------------------------------------------------
# converts timestamp to epoch (day,evening,night) pretty straightforward right?
def epochCalc(timestamps):
	splitTimes = unixTimeConv(timestamps)
	epochTimes = []
	for i in range(0,len(splitTimes)):
		hour=int(splitTimes[i,3])
		if hour >= 11 and hour <=21:
			epoch='day'
		else:
			epoch='night'

		epochTimes.append((epoch,splitTimes[i,6]))


	return epochTimes

#-----------------------------------------------------------------------
# computes average time between stress reports in order to find the optimal 
# time window for the app statistics calculation to reduce overlapping of features
def meanStress(cur,uid):
	records = sorted( loadStressLabels(cur,uid) , key=lambda x:x[0] )
	mean = 0 

	for i in range(0,len(records)-1):
		mean += records[i+1][0] - records[i][0]

	mean = float(mean) / len(records)
	return(mean)


#---------------------------------------------------------------------------------------
# counts occurences of bag-of-apps for given user 'uid' during experiment
# in the average report time window around 'timeQuery' stress report
def countAppOccur(cur,uid,timeQuery,timeW):
	cur.execute("SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp <= %s AND time_stamp>=%s;",[uid,timeQuery,timeQuery-timeW])

	#Counter class counts occurrences of unique ids
	records =Counter( cur.fetchall() )
	
	#transforming keys cause of ugly return shape of Counter class
	for k in records.keys():
		records[k[0]] = records.pop(k)

	return records



		


#---------------------------------------------------------------------------------------
# returns True if screen was On at given time, false otherwise
def checkScreenOn(cur,uid,time):
	uid = uid +'lock'
	time = int(time)
	cur.execute("SELECT * FROM {0} WHERE timeStart <= {1} AND timeStop >={2} ; ".format(uid,time,time) )
	records = cur.fetchall()

	if not records:
		return(False)
	else:
		return(True)


#---------------------------------------------------------------------------------------
# retrieves stress labels for user 'uid' and returns a list with their corresponding timestamps
def loadStressLabels(cur,uid):
	cur.execute("SELECT time_stamp,stress_level  FROM {0} ".format(uid) )
	records = cur.fetchall()

	return records	

#---------------------------------------------------------------------------------------
def loadMoodLabels(cur,uid):
	uid = uid+'mood'
	cur.execute("SELECT time_stamp, mood  FROM {0} ".format(uid) )
	records = cur.fetchall()

	return records	




# TODO: add features regarding screen on/off and phone lock

# Takes as input the list with all Feature Vectors and returns the train matrix X
# Bag-of-Apps approach is followed to construct the Train Matrix X 
# FVlist initially contains all FVs which have different length, BOA when returned
def constructBOA(FVlist):
	allkeys = []
	newList = []
	# after this loop 'allkeys' will hold all unique keys
	# TODO: might be poor in terms of performance, to check for possible optimizations
	for i in range(0,len(FVlist)):
		for key in FVlist[i].keys():
			if key not in allkeys:
				allkeys.append(key)
		#Each FV was a dictionary, transforming to SortedDict to later construct BOA matrix
		b=dict(FVlist[i])
		a = SortedDict(b)
		newList.append(a)

	Xtrain = np.empty([len(FVlist),len(allkeys)],dtype=int)

	# The final length of each FV is the unique apps that appeared. Zero is inserted in cell 
	# that corresponds to an app_id that did not occur during specific period
	for i in range(0,len(FVlist)):
		for key in allkeys:
			if key not in newList[i].keys():
				newList[i][key] = 0

	for i in range(0,len(Xtrain)):
		Xtrain[i] = newList[i].values()


	return(Xtrain)


# Takes as input Matrix with rows of features and picks out the most common apps (columns)
# The average application usage is computed and the applications with the best average are kept
def selectBestFeatures(X,mc):
	# average of each column is in 'av'
	toDel = X.shape[1] -mc
	for i in range(0,toDel):
		av = np.mean(X, axis=0)
		m = np.argmin(av)
		X = np.delete(X,m,1)
	return(X)


# Screen time features are computed to enhance the training procedure
# Possibly increase prediction accuracy
# returned list contains 7 features
def screenStatFeatures(cur,uid,timestamp,timeWin):
	featList = []
	for i in ['lock']:
		uidL= uid +i
		totalOn =0
		# Getting data phone lock data from appropriate tables to compute statistics of use
		cur.execute('SELECT * FROM {0} WHERE timeStart <= {1} AND timeStop >= {2}'.format(uidL,timestamp,timestamp-timeWin))
		screenTime = np.array(cur.fetchall())

		#Checking if there are ANY data in this period
		if screenTime.shape[0]!=0:
			# Total time is calculated as follows:
			#       last row,second cell - first row, first cell          
			totalTime = screenTime[-1][1] - screenTime[0][0]
			
			#instantiating arrays to hold times for locked/unlocked
			timeOn = np.empty(screenTime.shape[0])
			timeOff = np.empty(screenTime.shape[0])



			timeOn[0] = screenTime[0][1]-screenTime[0][0]
			totalOn += timeOn[0]
			# timeOn cells hold the total time phone remained unlocked
			# timeOff for the opposite
			for i in range(1,len(screenTime)):
				timeOn[i] = screenTime[i][1]-screenTime[i][0]
				totalOn += timeOn[i]

				timeOff[i] = screenTime[i][0] - screenTime[i-1][1]



			totalOff= totalTime -totalOn

			#computing and appending statistics to returned list
			#featList.extend(np.mean(timeOn),np.std(timeOn),np.var(timeOn),)
			
			featList.append(np.mean(timeOn))
			featList.append(np.std(timeOn))
			featList.append(np.var(timeOn))
			
			featList.append(np.mean(timeOff))
			featList.append(np.std(timeOff))
			featList.append(np.var(timeOff))

			featList.append(len(screenTime))
			featList.append(totalOn)
			featList.append(totalOff)

			featList.append(np.amax(timeOn))
			featList.append(np.amax(timeOff))

			# converting to np array for compatibility with other FVs
			#return(np.array(featList))

		else:
			featList.extend( np.zeros(11) )
	A= np.nan_to_num(featList)
	if np.std(A)>0:
		A = (A-np.mean(A))/np.std(A)
	return(A)

# Computes average number people(BT scans) for two periods in a day(first half and second half of day)
# Stats computed are always proceeding stress reports
def colocationStats(cur,uid,timestamp):
	meanCo = np.zeros(2)
	for i in [0,1]:
		total = 0

		cur.execute("SELECT time_stamp,mac FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2}".format(uid+'bt',timestamp-(i+1)*halfday,timestamp-i*halfday))
		records = cur.fetchall() 

		#By counting how many times each timestamp appeared, we get the number of nearby people
		times =[item[0] for item in records]
		#print(times)
		if len(set(times)) >0:
			uniqueTimes = list(set(times))

			for t in uniqueTimes:
				#print(times.count(t))
				total += times.count(t)
			#mean number of peo
			#print(total, len(times))
			meanCo[i] = float(total) / len(set(times))
	meanCo = np.nan_to_num(meanCo)
	return(meanCo)

def conversationStats(cur,uid,timestamp):
	totalConvTime=np.zeros(2)
	totalConvs = np.zeros(2)
	totalFeats = np.empty(10)
	for i in [0,1]:
		cur.execute('SELECT * FROM {0} WHERE start_timestamp >= {1} AND end_timestamp<= {2}'.format(uid+'con',timestamp-(i+1)*halfday,timestamp-i*halfday))
		records = cur.fetchall() 
		timeCon = np.empty(len(records))

		totalConvs[i] = len(records)
		#this is the TRUE power of python
		for j in range(0,len(records)):
			timeCon[j] = records[j][1]-records[j][0]

		totalConvTime[i] = sum([item[1]-item[0] for item in records])
		#print(totalConvTime[i])
	#print(np.std(timeCon),np.var(timeCon))
	
	a=np.concatenate((totalConvs,totalConvTime),axis=0)
	a=np.append(a, np.var(timeCon))
	a=np.append(a, np.std(timeCon))
	a=np.nan_to_num(a)
	if np.std(a)>0:
		a = (a-np.mean(a))/np.std(a)
	#print(a)
	#concatenate 4 features in one nparray before returning
	return(a)

#testing
#con = psycopg2.connect(database='dataset', user='tabrianos')
#cur = con.cursor()
#print(screenStatFeatures(cur,'u00',1365183210,meanStress(cur,'u00')))
#print(meanStress(cur,'u00'))
#t = 1366885867 
#t1 = t+1000
#t2= t1+5000
#print(unixTimeConv((t,t1,t2)))
#print(conversationStats(cur,'u00',t))


#print(colocationStats(cur,'u00',t ))
#d = countAppOccur(cur,'u59',30,t)
#loadStressLabels(cur,'u01')
#a=computeAppStats(cur,'u09',day)
#print(a[0][2])
#print(a[1][65])
#print(a[2])
#print((appTimeIntervals(cur,'u00',1366752858,day)))
#print(epochCalc(1234551100))
#countAppOccur(cur,'u01')
#num = 1365284453
#print(epochStressNorm(cur,'u41'))
#mapStressEpochs(cur)
#print(checkScreenOn(cur,'u00',num))


#[DONE]: function that produces labels [stressed/not stressed] from surveys [DONE]
#[DONE]: function that computes application usage statistics in time window (hour/day/week) (frequency)
#[DONE]: function that computes time intervals between subsequent app usages (not background, only user ) cross-checked with screen info


#TODO: migrate database to NoSQL																																																																																																																																																									
#TODO: function that computes sms+calls statistical features in time window (how many sms, how many people)
#NOTE: some call+sms logs do not contain any data (maybe corrupted download?)

