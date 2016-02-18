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
'u56','u57','u58']

#uids1=['u59','u57','u02','u52','u16','u19','u44','u24','u51','u00','u08']
#uids2=['u02','u00','u57']
uids2=['u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','u16','u19']
uids1=['u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','u16','u19']

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
		if hour >10 and hour <=18:
			epoch='day'
		elif hour >0 and hour<=10:
			epoch='night'
		else:
			epoch='evening'
		epochTimes.append((epoch,splitTimes[i,6]))
	return epochTimes

#-----------------------------------------------------------------------


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
	uid = uid+'mood2'
	cur.execute("SELECT time_stamp, mood  FROM {0} ".format(uid) )
	records = cur.fetchall()

	return records	





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
			#         last row,second cell - first row, first cell          
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



#------------------------------------------------------------------------------------------------
def colocationEpochFeats(cur,uid,timestamp):
	"""Calculates total number of devices around user at hour intervals for previous day
	"""
	j=0
	hour=3600
	nearby_dev = np.zeros(23)
	closer_dev = np.zeros(23)
	for i in range(1,24):

		cur.execute("SELECT time_stamp,mac,level FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2}"
			.format(uid+'bt',timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))

		records = cur.fetchall()
		# In every BT scan all MACs share the same timestamp, thus the number of MACs
		# at each given time reveals the number of nearby devices which we assume 
		# is positivelY correlated with the number of humans around the user.
		# A distinction between nearby and closer-to-user devices is being made
		# based on signal strength threshold
		times_near = [item[0] for item in records if item[2]<-80]
		nearby_dev[i-1] = len(times_near)

		times_closer = [item[0] for item in records if item[2]>=-80]
		closer_dev[i-1] = len(times_closer)
		
	bt_feats = np.hstack((closer_dev,nearby_dev))
	return(bt_feats)


#------------------------------------------------------------------------------------------------





def convEpochFeats(cur,uid,timestamp):
	"""Returns total duration and number of conversations
	   calculated in ones hour intervals"""

	hour = 3600
	totalConvTime = np.zeros(23)
	totalConvs = np.zeros(23)
	for i in range(1,24):
		cur.execute('SELECT * FROM {0} WHERE start_timestamp >= {1} AND end_timestamp<= {2}'
			.format(uid+'con',timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))

		records = cur.fetchall()
		# Sum over the duration of all conversation in that hour
		totalConvTime[i-1] = sum([(records[i][1]-records[i][0]) for i in range(0,len(records))])
		# Count their total number
		totalConvs[i-1] = len(records)
	
	# Concatenate to one row before returning
	return(np.hstack((totalConvTime,totalConvs)))




def activityEpochFeats(cur,uid,timestamp):
	"""Returns stationary to moving ratio,std and variance of activity
	   inferences in one hour intervals
	"""
	statToMovingRatio = np.zeros(23)
	var_stats = np.zeros(23)
	std_stats = np.zeros(23)

	uidS = uid +'act'
	for i in range(1,24):
		# Retrieving data in hour intervals spanning over one day
		cur.execute('SELECT activity FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'
			.format(uidS,timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))
		records = cur.fetchall()

		var_stats[i-1] = np.var(records)
		std_stats[i-1] = np.std(records)

		# Calculating number of stationary and walking/running occurences
		stationary = len([item for item in records if item==0])
		moving = len([item for item in records if item==1 or item==2])

		if moving>0:
			statToMovingRatio[i-1] = float(stationary) / moving
		else:
			statToMovingRatio[i-1] = 0

	return(np.nan_to_num(np.hstack((statToMovingRatio,var_stats,std_stats))))




def audioEpochFeats(cur,uid,timestamp):
	""" Returns voice to silence ratio, total noise occurences, std,variance 
		in one hour intervals one day prior to report
	"""
	uidA = uid +'audio'

	var_stats = np.zeros(23)
	std_stats = np.zeros(23)
	voiceToSilenceRatio = np.zeros(23)

	for i in range(1,24):
		cur.execute('SELECT audio FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'
			.format(uidA,timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))
		records = cur.fetchall()

		var_stats[i-1] = np.var(records)
		std_stats[i-1] = np.std(records)

		# Calculating number of silence and voice/noise occurences
		silence = len([item for item in records if item==0])
		voice = len([item for item in records if item==1 or item==2])

		if silence>0:
			voiceToSilenceRatio[i-1] = float(voice) / silence
		else:
			voiceToSilenceRatio[i-1] = 0

	return(np.nan_to_num(np.hstack((voiceToSilenceRatio,var_stats,std_stats))))


def gpsFeats(cur,uid,timestamp,centers):
	# number of clusters as defined by DBSCAN: 14 + 1 for out of town
	# p will hold the percentage of time spent during previous day in each cluster 
	p = np.zeros(15)

	#variances = np.zeros(2)
	cur.execute("SELECT time_stamp,latitude,longitude FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2} AND travelstate=0".format(uid+'gpsdata',timestamp-day,timestamp))
	records = cur.fetchall()

	if not records:
		return(np.zeros(1))

	for i in range(0,len(records)):
		#print(records[i][1],records[i][2])
		# if user is in campus assign him to one of 14 clusters
		# otherwise assign to 15th cluster which stands for 'out-of-town'
		if (records[i][1] > 43.60 and records[i][1] <43.75 and records[i][2] > -72.35 and records[i][2] < -72.2):
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
	featureVector = np.nan_to_num(np.array([e]))

	return featureVector


def my_greatcircle(a,b):
	return(great_circle(a,b).meters)



def main():
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()
	#warnings.simplefilter("error")
	centers = np.load('visualizations/clustercenters.npy')

# ------------TEST CASE-----------------------------
	for loso in uids1:
		ytest=[]
		accuracies =[]
		acc=0
		maxminAcc =[]
		Xbig = np.zeros([1,242])	
		Ybig = np.zeros([1])
		labels=[]
		labels.append(19)
		# loso means leave one student out: forest is trained on other users data
		# then tests are run on 'loso' student 
		uids2.remove(loso)
		uids2.append(loso)
		print('LOSO: {0}'.format(loso))
		for testUser in uids2:
			print(testUser)
			# lists that temporary store features before concatenation
			ScreenList = []
			colocationList =[]
			conversationList =[]
			activityList=[]
			audioList = []
			gpsList = []

			# loading stress labels from database (currently on 0-5 scale)
			records = loadStressLabels(cur,testUser) 
		

			
			#X,Y store initially the dataset and the labels accordingly
			Y = np.zeros(len(records))
			X = np.array(records)

	


			for i in range(0,len(records)):
				print(i)
				colocationList.append( colocationEpochFeats(cur,testUser,X[i][0]))
				conversationList.append( convEpochFeats(cur,testUser,X[i][0]))
				activityList.append(activityEpochFeats(cur,testUser,X[i][0]))
				ScreenList.append( screenStatFeatures(cur,testUser,X[i][0],day) )
				audioList.append(audioEpochFeats(cur,testUser,X[i][0]))
				gpsList.append(gpsFeats(cur,testUser,X[i][0],centers))
				#print(gpsFeats(cur,testUser,X[i][0],centers))
				if testUser==loso:
					ytest.append(X[i][1])
				#labels list holds user ids to be used in LeaveOneOut pipeline
				labels.append(testUser[-2:])
				Y[i] = X[i][1]

			
			#concatenating features in one array 

			temp_gps = np.reshape(np.array(gpsList,dtype=np.float64), (len(np.array(gpsList)),1))
			Xtt = np.concatenate((np.array(activityList),np.array(ScreenList),np.array(conversationList),np.array(colocationList),np.array(audioList),temp_gps),axis=1)
			print(Xtt.shape)

			#initiating and training forest, n_jobs indicates threads, -1 means all available
			# while the test student is not reached, training data are merged into one big matrix
			Xbig = np.concatenate((Xbig,Xtt),axis=0)
			Ybig = np.concatenate((Ybig,Y),axis=0)

			del ScreenList[:]
			del colocationList[:]
			del conversationList[:]
			del activityList[:]
			del audioList[:]
			del gpsList[:]



			if testUser!=loso:
				Xbig = Xbig.astype(np.float64)
				print(Xbig.dtype)
				

			# when loso, tests are run
			elif testUser==loso:
				#Xbig = preprocessing.scale(Xbig)
				np.save('numdata/withgps/hourlyFeats.npy',Xbig)
				np.save('numdata/withgps/hourlyLabels.npy',Ybig)
				np.save('numdata/withgps/hourlyLOO.npy',np.array(labels))
				print(Xbig.shape[0],Ybig.shape[0],len(labels))
				print('train matrix saved')
				a = raw_input()
				forest = RandomForestClassifier(n_estimators=100, n_jobs = -1)
				forest.fit(Xbig,Ybig)
				ef = forest.score(Xtt,ytest)
				print(ef*100)

				output = np.array(forest.predict(Xtt))
				scored = output - np.array(ytest)

				# Counting as correct predictions the ones which fall in +/-1, not only exact
				# I call it the 'Tolerance technique'
				correct=0
				c = Counter(scored)
				for k in c.keys():
					if k<2 and k>-2:
						correct += c[k]
				
				score = float(correct)/len(scored)
				print(score*100)



		print(Xbig.shape)
	
		



if __name__ == '__main__':
	main()






#testing
#con = psycopg2.connect(database='dataset', user='tabrianos')
#cur = con.cursor()
#centers = np.load('visualizations/clustercenters.npy')
#t1 = 1368481065
#print(gpsFeats(cur,'u19',t,centers))
#print(len(screenStatFeatures(cur,'u00',t1,day)))
#print(meanStress(cur,'u00'))
#t1= 1365111111
#print(len(colocationEpochFeats(cur,'u02',t1)))
#print(len(convEpochFeats(cur,'u00',t1)))
#print(len(activityEpochFeats(cur,'u59',t1)))
#print(len(conversationStats(cur,'u00',t)))
#print(len(audioEpochFeats(cur,'u00',t1)))
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

