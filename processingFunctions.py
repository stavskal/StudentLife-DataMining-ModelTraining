import datetime,psycopg2
from collections import Counter
import numpy as np
import math
from sortedcontainers import SortedDict
from sklearn import preprocessing
from geopy.distance import great_circle
from scipy.spatial.distance import euclidean
from scipy.stats import entropy

#---------------------------------------------
#This script contains a collection of functions
#that are useful in processing the data in
#StudentLife Dataset

# Each functions has its description and internal comments
# follow the code as well.
#---------------------------------------------


# time lengths expresses in Seconds
hour = 3600
day = 86400
halfday = 43200
weekSec = 604000


# List of all users in dataset
uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

# List of 'good' users
#uids1=['u00','u02','u12','u24','u08','u57','u52','u51','u59']
uids1=['u16','u19','u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49']


#---------------------------------------------------------------------------------------

def unixTimeConv(timestamps):
	""" converts unix timestamp to human readable date 
		(e.g '1234567890' -> '2009 02 14  00 31 30')
	"""
	splitTimes = np.zeros((len(timestamps), 7),dtype='float32')
	i=0
	#print(timestamps)
	for time in timestamps:
		newTime = str(datetime.datetime.fromtimestamp(int(time)))
		yearDate,timeT = newTime.split(' ')
		year,month,day = str(yearDate).split('-')
		hour,minutes,sec = timeT.split(':')
		splitTimes[i,:] = (year,month,day,hour,minutes,sec,time)
		i += 1
	return(splitTimes)

#----------------------------------------------------------------------------------------

def epochCalc(timestamps):
	""" converts timestamp to epoch (day,evening,night) 
		and returns (epoch,time) tuple
	"""
	splitTimes = unixTimeConv(timestamps)
	epochTimes = []
	for i in range(0,len(splitTimes)):
		hour=int(splitTimes[i,3])
	#	print(hour)
		if hour >9 and hour <=18:
			epoch='day'
		elif hour >0 and hour<=9:
			epoch='night'
		else:
			epoch='evening'
		epochTimes.append((epoch,splitTimes[i,6]))
	return epochTimes

#-----------------------------------------------------------------------

def meanStress(cur,uid):
	""" computes average time between stress reports in order to find the optimal 
	    time window for the app statistics calculation to reduce overlapping of features
	"""
	records = sorted( loadStressLabels(cur,uid) , key=lambda x:x[0] )
	mean = 0 

	for i in range(0,len(records)-1):
		mean += records[i+1][0] - records[i][0]

	mean = float(mean) / len(records)
	return(mean)


#---------------------------------------------------------------------------------------

def countAppOccur(cur,uid,timeQuery,timeW):
	""" counts occurences of bag-of-apps for given user 'uid' during experiment
	    in the average report time window around 'timeQuery' stress report
	"""
	cur.execute("SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp <= %s AND time_stamp>=%s;",[uid,timeQuery,timeQuery-timeW])

	#Counter class counts occurrences of unique ids
	records =Counter( cur.fetchall() )
	
	#transforming keys cause of ugly return shape of Counter class
	for k in records.keys():
		records[k[0]] = records.pop(k)

	return records



#---------------------------------------------------------------------------------------

def checkScreenOn(cur,uid,time):
	""" returns True if screen was On at given time, false otherwise
	"""
	uid = uid +'lock'
	time = int(time)
	cur.execute("SELECT * FROM {0} WHERE timeStart <= {1} AND timeStop >={2} ; ".format(uid,time,time) )
	records = cur.fetchall()

	if not records:
		return(False)
	else:
		return(True)


#---------------------------------------------------------------------------------------

def loadStressLabels(cur,uid):
	"""retrieves stress labels for user 'uid' and returns a list with their corresponding timestamps
	"""
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


# Takes as input Matrix with rows of features and picks out the most common features (columns), filters
# out the ones that occur the least
# The average application usage is computed and the applications with the best average are kept
def selectBestFeatures(X,mc):
	# average of each column is in 'av'
	toDel = X.shape[1] -mc
	for i in range(0,toDel):
		av = np.mean(X, axis=0)
		m = np.argmin(av)
		X = np.delete(X,m,1)
	return(X)



def screenStatFeatures(cur,uid,timestamp,timeWin):
	""" Screen time features are computed to enhance the training procedure
		returned list contains 7 features
	"""
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

		else:
			featList.extend( np.zeros(11) )
	A= np.nan_to_num(featList)
	A=preprocessing.scale(A)
	return(A)




def colocationEpochFeats(cur,uid,timestamp):
	"""Calculates total and average number of people around user at three epochs, 6 features total
	"""
	j=0
	colocFeats = np.zeros(6)
	cur.execute("SELECT time_stamp,mac FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2}".format(uid+'bt',timestamp-2*halfday,timestamp))
	records = cur.fetchall() 

	times =[item[0] for item in records]
	timeEpochs = epochCalc(times)

	#for every epoch count the total number of people around
	for ep in ['day','evening','night']:
		timesE = [item[1] for item in timeEpochs if item[0]==ep ]
		
		if len(set(timesE)) >0:
			uniqueTimes = list(set(timesE))

			for t in uniqueTimes:
				colocFeats[j] += timesE.count(t)

			colocFeats[j+1] = float(colocFeats[j])/ len(set(timesE))

		#step is +2 because 2 features are calculated for every epoch, total number and average
		j += 2 
	return(colocFeats)






def convEpochFeats(cur,uid,timestamp):
	"""Returns total duration and number of conversations
	   calculated in three epochs (day,evening,night), 6 features total"""

	cur.execute('SELECT * FROM {0} WHERE start_timestamp >= {1} AND end_timestamp<= {2}'.format(uid+'con',timestamp-day,timestamp))
	records = cur.fetchall()

	totalConvsEvening=0
	totalConvsDay=0
	totalConvsNight=0

	totalConvTimeE=0
	totalConvTimeD = 0
	totalConvTimeN=0

	tStart = [item[0] for item in records]
	tStop = [item[1] for item in records]

	timeEpochs = epochCalc(tStart)
	timeEpochs1 = epochCalc(tStop)

	for i in range(0,len(records)):
		if timeEpochs[i][0] in ['evening']:
			totalConvsEvening += 1 
			totalConvTimeE += records[i][1]-records[i][0]

		if timeEpochs[i][0] in ['day']:
			totalConvsDay += 1 
			totalConvTimeD += records[i][1]-records[i][0]

		if timeEpochs[i][0] in ['night']:
			totalConvsNight += 1 
			totalConvTimeN += records[i][1]-records[i][0]
	
	# concatenating all variables into FV vector		
	FV = np.array((totalConvsEvening,totalConvsNight,totalConvsDay,totalConvTimeN,totalConvTimeD,totalConvTimeE))
	return(FV)




def activityEpochFeats(cur,uid,timestamp):
	"""Returns stationary to moving ratio in three epochs, 3 features total
	"""
	totalDur = 0
	statToMovingRatio = np.zeros(3)
	stationary = np.zeros(3)
	moving = np.zeros(3)

	#retrieving data
	uidS = uid +'act'
	cur.execute('SELECT time_stamp,activity FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'.format(uidS,timestamp-2*halfday,timestamp))
	records = cur.fetchall()

	tStart = [item[0] for item in records]
	timeEpochs = (epochCalc(tStart))

	# Scaning all records, if timestamp belongs to certain epoch then 
	# the corresponding cell in stationary/moving arrays is incremented 
	for i in range(0,len(records)):
		if timeEpochs[i][0]=='day':
			if records[i][1]==0:
				stationary[0] += 1
			else:
				moving[0] += 1

		if timeEpochs[i][0]=='evening':
			if records[i][1]==0:
				stationary[1] += 1
			else:
				moving[1] += 1

		if timeEpochs[i][0]=='night':
			if records[i][1]==0:
				stationary[2] += 1
			else:
				moving[2] += 1

	for i in range(0,3):
		if moving[i]>0:
			statToMovingRatio[i] = float(stationary[i]) / moving[i]
		else:
			statToMovingRatio[i] = 0

	return(statToMovingRatio)




def audioEpochFeats(cur,uid,timestamp):
	""" Returns voice to silence ratio and total noise occurences in three epochs
		one day prior to report
	"""
	uidA = uid +'audio'
	
	silence = np.zeros(3)
	noise = np.zeros(3)
	voice = np.zeros(3)

	voiceToSilenceRatio = np.zeros(3)

	cur.execute('SELECT time_stamp, audio FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'.format(uidA,timestamp-2*halfday,timestamp))
	records = cur.fetchall()

	tStart = [item[0] for item in records if item[1]!=3]
	timeEpochs = (epochCalc(tStart))

	#counting occurences of each class in each epoch
	for i in range(0,len(records)):
		if timeEpochs[i][0]=='day':
			if records[i][1]==0:
				silence[0] += 1
			elif records[i][1]==1:
				voice[0] += 1
			else:
				noise[0] +=1

		if timeEpochs[i][0]=='evening':
			if records[i][1]==0:
				silence[1] += 1
			elif records[i][1]==1:
				voice[1] += 1
			else:
				noise[1] +=1

		if timeEpochs[i][0]=='night':
			if records[i][1]==0:
				silence[2] += 1
			elif records[i][1]==1:
				voice[2] += 1
			else:
				noise[2] +=1

	for i in range(0,3):
		if silence[i]>0:
			voiceToSilenceRatio[i] = float(voice[i]) / silence[i]

	return(np.concatenate((voiceToSilenceRatio,noise),axis=0))


# NOT TESTED YET
def gpsFeats(cur,uid,timestamp,centers):
	# number of clusters as defined by DBSCAN: 14 + 1 for out of town
	# p will hold the percentage of time spent during previous day in each cluster 
	p = np.zeros(15)

	variances = np.zeros(2)
	cur.execute("SELECT time_stamp,latitude,longitude FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2} AND travelstate=0".format(uid+'gpsdata',timestamp-day,timestamp))
	records = cur.fetchall()

	# variance of latitudes and longitudes
	variances[0] = np.var([i[1] for i in records])
	variances[1] = np.var([i[2] for i in records])

	locationVar = np.log(variances[0] + variances[1])

	for i in range(0,len(records)):
		# if user is in campus assign him to one of 14 clusters
		# otherwise assign to 15th cluster which stands for 'Out-of-town'
		if records[i][1] > 43.60 and records[i][1] <43.75:
			if records[i][2] > -72.35 and records[i][2] < -72.2:
				# for every gps coordinate pair calculate the distance from cluster
				# centers and assign to the nearest
				distFromCenters = np.apply_along_axis(my_greatcircle,1,centers,np.array(records[i][1:3]))
				mindist = np.argmin(distFromCenters)
				p[mindist] += 1
			
		else:
			# student is out of town
			p[14] += 1

	#calculating GPS entropy
	e = entropy(p/float(sum(p)))
	featureVector = np.array([[e,variances[0],variances[1]]])
	print(featureVector)
	return e


def my_greatcircle(a,b):
	return(great_circle(a,b).meters)

"""
def gpsEntropyFeat(centers,gpscoords):
	# number of clusters as defined by DBSCAN: 14
	# p will hold the percentage of time spent during previous
	# day in each cluster 
	p = np.zeros(14)

	# for every gps coordinate pair calculate the distance from cluster
	# centers and assign to the nearest one
	for coordpair in gpscoords:
		distances = np.apply_along_axis(my_greatcircle,1,centers,coordpair)
		mindist = np.argmin(distances)
		p(mindist) += 1

"""




#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
centers = np.load('visualizations/clustercenters.npy')
t = 1368481065

gpsFeats(cur,'u19',t,centers)
#print(screenStatFeatures(cur,'u00',1365183210,meanStress(cur,'u00')))
#print(meanStress(cur,'u00'))
#t1= 1365111111
#print(colocationEpochFeats(cur,'u00',t1))
#print(convEpochFeats(cur,'u00',t))
#print(activityEpochFeats(cur,'u00',t))
#print(conversationStats(cur,'u00',t))
#print(audioEpochFeats(cur,'u00',t))
#print(gcd(43.7066671,-72.2890974,  43.7067476, -72.2892027))
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


#TODO: migrate database to NoSQL																																																																																																																																																									

