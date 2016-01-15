import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time
import warnings



day = 86400
halfday = 43200
quarterday = 21600

times =[2*day]

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58']

#uids1=['u59','u57','u02','u52','u16','u19','u44','u24','u51','u00','u08']
#uids2=['u02','u00','u57']
uids2=['u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','u16','u19']

ch = [120,100,70,50,35]



# returns feature vector corresponing to (timestamp,stress_level) (report)
# This feature vector is of size mc(=Most Common), which varies due to Cross Validation.
# each cell corresponds to the % of usage for each app. Apps that were not used during 
# previous day have zero in feature vector cell
def appStatsL(cur,uid,timestamp,timeWin):
	
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp <= %s AND time_stamp >= %s; """, [uid,timestamp,timestamp-day] )
	records= Counter( cur.fetchall() )

	for k in records.keys():
		records[k[0]] = records.pop(k)

	return records


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
		Xbig = np.zeros([1,51])	
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
				np.save('numdata/withgps/epochFeats.npy',Xbig)
				np.save('numdata/withgps/epochLabels.npy',Ybig)
				np.save('numdata/withgps/LOO.npy',np.array(labels))
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