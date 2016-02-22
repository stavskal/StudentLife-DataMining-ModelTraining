import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from sklearn import preprocessing
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time
import warnings



day = 86400
halfday = 43200
quarterday = 21600
hour = 3600
hour3 = 10800
times =[2*day]

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58']

#uids1=['u59','u57','u02','u52','u16','u19','u44','u24','u51','u00','u08']
#uids2=['u02','u00','u57']
uids2=['u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','u16','u19']

ch = [120,100,70,50,35]

def loadSleepLabels(cur,uid):
	uid = uid+'sleep'

	cur.execute('SELECT hour,time_stamp,rate FROM {0}'.format(uid))
	records = cur.fetchall()
	#records = sorted(records,key=lambda x:x[1])
	return(np.array(records)) 	

def unixTimeConv(timestamps):
	""" converts unix timestamp to human readable date 
		(e.g '1234567890' -> '2009 02 14  00 31 30')
	"""

	newTime = str(datetime.datetime.fromtimestamp(int(timestamps)))
	yearDate,timeT = newTime.split(' ')
	year,month,day = str(yearDate).split('-')
	hour,minutes,sec = timeT.split(':')
	splitTimes = (year,month,day,hour,minutes,sec,timestamps)

	return(splitTimes)

def epochCalc(timestamps):
	""" converts timestamp to epoch (day,evening,night) 
		and returns (epoch,time) tuple
	"""
	splitTimes = unixTimeConv(timestamps)
	epochTimes = []
	hour=int(splitTimes[3])

	if (hour >0 and hour <=9) or hour>=23:
		epoch='night'
	else:
		epoch='not_night'
	epochTimes.append((epoch,splitTimes[6]))
	return epochTimes


def colocationEpochFeats(cur,uid,timestamp):
	"""Calculates total and average number of people around user at three epochs, 6 features total
	"""
	j=0
	hour=3600
	nearby_dev = []
	closer_dev = []
	for i in range(1,8):
		hs_timestamp = timestamp-86400+(i-1)*hour3
		he_timestamp = timestamp-86400+i*hour3
		# Determining if start/end time of given hour is in the night
		# If yes, proceed with feature calculation, if not skip
		s_epoch = epochCalc(hs_timestamp)
		e_epoch = epochCalc(he_timestamp)

		if s_epoch[0][0]=='night' or e_epoch[0][0]=='night':

			cur.execute("SELECT time_stamp,mac,level FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2}"
				.format(uid+'bt',hs_timestamp,he_timestamp))

			records = cur.fetchall()
			# In every BT scan all MACs share the same timestamp, thus the number of MACs
			# at each given time reveals the number of nearby devices which we assume 
			# is positively correlated with the number of humans around the user.
			# A distinction between nearby and closer-to-user devices is being made
			# based on signal strength threshold
			times_near = [item[1] for item in records if item[2]<-80]
			nearby_dev.append( len(set(times_near)))

			times_closer = [item[1] for item in records if item[2]>=-80]
			closer_dev.append(len(set(times_closer)))

	bt_feats = np.hstack((closer_dev,nearby_dev))
	return(bt_feats)

def convEpochFeats(cur,uid,timestamp):
	"""Returns total duration and number of conversations
	   calculated in ones hour intervals"""

	hour = 3600
	totalConvTime = []
	totalConvs = []
	for i in range(1,8):
		hs_timestamp = timestamp-86400+(i-1)*hour3
		he_timestamp = timestamp-86400+i*hour3
		# Determining if start/end time of given hour is in the night
		# If yes, proceed with feature calculation, if not skip
		s_epoch = epochCalc(hs_timestamp)
		e_epoch = epochCalc(he_timestamp)
	
		if s_epoch[0][0]=='night' or e_epoch[0][0]=='night':
			cur.execute('SELECT * FROM {0} WHERE start_timestamp >= {1} AND end_timestamp<= {2}'
				.format(uid+'con',timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))

			records = cur.fetchall()
			# Sum over the duration of all conversation in that hour
			totalConvTime.append( sum([(records[i][1]-records[i][0]) for i in range(0,len(records))]))
			# Count their total number
			totalConvs.append(len(records))
	# Concatenate to one row before returning
	return(np.hstack((totalConvTime,totalConvs)))

def activityEpochFeats(cur,uid,timestamp):
	"""Returns stationary to moving ratio,std and variance of activity
	   inferences in one hour intervals
	"""
	statToMovingRatio = []
	var_stats = []
	std_stats = []
	uidS = uid +'act'
	for i in range(1,8):
		hs_timestamp = timestamp-86400+(i-1)*hour3
		he_timestamp = timestamp-86400+i*hour3
		# Determining if start/end time of given hour is in the night
		# If yes, proceed with feature calculation, if not skip
		s_epoch = epochCalc(hs_timestamp)
		e_epoch = epochCalc(he_timestamp)

		if s_epoch[0][0]=='night' or e_epoch[0][0]=='night':
			# Retrieving data in hour intervals spanning over one day
			cur.execute('SELECT activity FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'
				.format(uidS,timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))
			records = cur.fetchall()

			var_stats.append(np.var(records))
			std_stats.append(np.std(records))

			# Calculating number of stationary and walking/running occurences
			stationary = len([item for item in records if item==0])
			moving = len([item for item in records if item==1 or item==2])

			if moving>0:
				statToMovingRatio.append(float(stationary) / moving)
			else:
				statToMovingRatio.append(0)
	return(np.nan_to_num(np.hstack((statToMovingRatio,var_stats,std_stats))))


def audioEpochFeats(cur,uid,timestamp):
	""" Returns voice to silence ratio, total noise occurences, std,variance 
		in one hour intervals one day prior to report
	"""
	uidA = uid +'audio'

	var_stats = []
	std_stats = []
	noise = []
	voiceToSilenceRatio = []

	for i in range(1,24):
		hs_timestamp = timestamp-86400+(i-1)*hour
		he_timestamp = timestamp-86400+i*hour
		# Determining if start/end time of given hour is in the night
		# If yes, proceed with feature calculation, if not skip
		s_epoch = epochCalc(hs_timestamp)
		e_epoch = epochCalc(he_timestamp)

		if s_epoch[0][0]=='night' or e_epoch[0][0]=='night':
			cur.execute('SELECT audio FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'
				.format(uidA,timestamp-86400+(i-1)*hour,timestamp-86400+i*hour))
			records = cur.fetchall()

			var_stats.append(np.var(records))
			std_stats.append(np.std(records))

			# Calculating number of silence and voice/noise occurences
			silence = len([item for item in records if item==0])
			voice = len([item for item in records if item==1 or item==2])
			noise.append(len([item for item in records if item==3]))
			if silence>0:
				voiceToSilenceRatio.append(float(voice) / silence)
			else:
				voiceToSilenceRatio.append(0)
	return(np.nan_to_num(np.hstack((voiceToSilenceRatio,var_stats,std_stats,noise))))
	"""
def main():
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()
	#warnings.simplefilter("error")
	#centers = np.load('visualizations/clustercenters.npy')

# ------------TEST CASE-----------------------------
	for loso in uids1:
		ytest=[]
		accuracies =[]
		acc=0
		maxminAcc =[]
		Xbig = np.zeros([1,132])	
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
			
			colocationList =[]
			conversationList =[]
			activityList=[]
			audioList = []

			# loading stress labels from database (currently on 0-5 scale)
			records = loadSleepLabels(cur,testUser) 
		

			
			#X,Y store initially the dataset and the labels accordingly
			Y = np.zeros(len(records))
			X = np.array(records)

	


			for i in range(0,len(records)):
				colocationList.append( colocationEpochFeats(cur,testUser,X[i][0]))
				conversationList.append( convEpochFeats(cur,testUser,X[i][0]))
				activityList.append(activityEpochFeats(cur,testUser,X[i][0]))
			#	ScreenList.append( screenStatFeatures(cur,testUser,X[i][0],day) )
				audioList.append(audioEpochFeats(cur,testUser,X[i][0]))
		
				if testUser==loso:
					ytest.append(X[i][1])
				#labels list holds user ids to be used in LeaveOneOut pipeline
				labels.append(testUser[-2:])
				Y[i] = X[i][2]

			
			#concatenating features in one array 

			Xtt = np.concatenate((np.array(activityList),np.array(conversationList),np.array(colocationList),np.array(audioList)),axis=1)
			print(Xtt.shape)

			#initiating and training forest, n_jobs indicates threads, -1 means all available
			# while the test student is not reached, training data are merged into one big matrix
			Xbig = np.concatenate((Xbig,Xtt),axis=0)
			Ybig = np.concatenate((Ybig,Y),axis=0)

			del colocationList[:]
			del conversationList[:]
			del activityList[:]
			del audioList[:]



			if testUser!=loso:
				Xbig = Xbig.astype(np.float64)
				print(Xbig.dtype)
				

			# when loso, tests are run
			elif testUser==loso:
				#Xbig = preprocessing.scale(Xbig)
				np.save('numdata/withgps/sleephourlyX.npy',Xbig)
				np.save('numdata/withgps/sleephourlyY.npy',Ybig)
				np.save('numdata/withgps/sleephourlyLOO.npy',np.array(labels))
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



	"""


#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
#centers = np.load('visualizations/clustercenters.npy')
t = 1368281065
print(len(convEpochFeats(cur,'u02',t)))
print(len(activityEpochFeats(cur,'u02',t)))
print(len(audioEpochFeats(cur,'u02',t)))
print(len(colocationEpochFeats(cur,'u00',t)))
#print(screenStatFeatures(cur,'u00',1365183210,meanStress(cur,'u00')))
#print(meanStress(cur,'u00'))
#t1= 1365111111
#rate = []
#rate.append(0)
#for u in uids2:
#	records = loadSleepLabels(cur,u) 
#	rate += [item[0] for item in records]
#	print(rate)
#print(len(rate),np.array(rate).shape)
#np.save('data/sleephourly_hours.npy',np.array(rate))
