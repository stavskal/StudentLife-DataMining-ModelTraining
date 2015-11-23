import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
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

uids1=['u12','u24','u08','u52','u51','u59','u57','u00','u02']

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
#testing
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()
	#warnings.simplefilter("error")

# ------------TEST CASE-----------------------------

	ytest=[]
	accuracies =[]
	acc=0
	totalP=0
	totalR=0
	maxminAcc =[]
	Xbig = np.zeros([1,19])	
	Ybig = np.ones([1])
	loso='u02'

	for testUser in uids1:
		print(testUser)
		Xlist = []
		ScreenList = []
		colocationList =[]
		conversationList =[]
		activityList=[]
		records = loadStressLabels(cur,testUser) 
	

		
		#X,Y store initially the dataset and the labels accordingly
		Y = np.empty(len(records))
		X = np.array(records)

		# X is shuffled twice to ensure that the report sequence is close to random
		#np.random.shuffle(X)
		#np.random.shuffle(X)


		for i in range(0,len(records)):
			colocationList.append( colocationStats(cur,testUser,X[i][0]))
			conversationList.append( conversationStats(cur,testUser,X[i][0]))
			activityList.append(activityFeats(cur,testUser,X[i][0]))
			#Xlist.append( appStatsL(cur,testUser,X[i][0],day) )
			ScreenList.append( screenStatFeatures(cur,testUser,X[i][0],day) )

			if testUser==loso:
				ytest.append(X[i][1])

			Y[i] = X[i][1]

		

		
		Xtt = np.concatenate((np.array(activityList),np.array(ScreenList),np.array(conversationList),np.array(colocationList)),axis=1)
		print(Xtt.shape)
		#initiating and training forest, n_jobs indicates threads, -1 means all available
		if testUser!=loso:
			Xbig = np.concatenate((Xbig,Xtt),axis=0)
			Ybig = np.concatenate((Ybig,Y),axis=0)
			
			Xbig = Xbig.astype(np.float64)
			forest = RandomForestClassifier(n_estimators=100, n_jobs = -1)
			forest.fit(Xbig,Ybig)
			print('forest done')

		elif testUser==loso:
			ef = forest.score(Xtt,Y)

			output = np.array(forest.predict(Xtt))
			scored = output - np.array(ytest)

			#Counting as correct predictions the ones which fall in +/-1
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