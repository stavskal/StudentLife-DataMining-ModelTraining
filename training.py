import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import time



day = 86400
halfday = 43200
quarterday = 21600

times =[2*day]

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

#uids1=['u00','u12','u19','u46','u59','u52','u57','u59','u08']
uids1=['u16','u19','u44','u24','u08','u51','u59','u57','u00','u02','u52']

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

# ------------TEST CASE-----------------------------
# A few users were picked from the dataset
# 70% of their stress reports and the corresponding features are used for training
# the rest 30% is used for testing. The train/test reports are randomly distributed
# throughout the whole experiment duration. No FV is used both for training and testing.
# After the 10 models are trained and tested, the overall accuracy is averaged
# Random Forests were picked due to their 'universal applicability', each with 25 decision trees


#TODO: maybe stick to a fixed number of apps and add more features such as screen on/off time(s), no of unique apps etc
	
	accuracies =[]
	acc=0
	totalP=0
	totalR=0
	maxminAcc =[]	
	
	for testUser in uids1:
		Xlist = []
		activityList = []
		ScreenList = []
		colocationList =[]
		conversationList =[]
		#cur.execute("SELECT time_stamp,stress_level FROM {0}".format(testUser))
		#records = cur.fetchall()

		records = loadStressLabels(cur,testUser)

		#meanTime = meanStress(cur,testUser)
 
	

		
		#X,Y store initially the dataset and the labels accordingly
		Y = np.zeros(len(records))
		X = np.array(records)

		# X is shuffled twice to ensure that the report sequence is close to random
		#np.random.shuffle(X)
		#np.random.shuffle(X)

		# Xlist contains Feature Vectors for Applications of different lengths according to each period
		# ScreenList contains FVs regarding screen info, fixed length (=7) for same periods
		t0 = time.time()
		for i in range(0,len(records)):
			colocationList.append( colocationStats(cur,testUser,X[i][0]) )
			conversationList.append( convEpochFeats(cur,testUser,X[i][0]) 	)
			activityList.append( activityEpochFeats(cur,testUser,X[i][0])  )
			ScreenList.append( screenStatFeatures(cur,testUser,X[i][0],day) )
			Y[i] = X[i][1]

		
		t1 = time.time()


		Xtt = np.concatenate((np.array(activityList),np.array(ScreenList),np.array(conversationList),np.array(colocationList)),axis=1)
		#print(Xtt[1,:])

		#print(Xtt)

		

		#initiating and training forest, n_jobs indicates threads, -1 means all available
		forest = RandomForestClassifier(n_estimators=100, n_jobs = -1)

		score = 0
		folds=3
		# Ensuring label percentage balance when K-folding
		skf = StratifiedKFold(Y, n_folds=folds)
		for train_index,test_index in skf:
			Xtrain,Xtest = Xtt[train_index], Xtt[test_index]
			ytrain,ytest = Y[train_index], Y[test_index]
			
			Xtrain = np.array(Xtrain,dtype='float64')
			Xtest = np.array(Xtest,dtype='float64')

			forest = forest.fit(Xtrain,ytrain)
			#score += forest.score(Xtest,ytest)
			output = np.array(forest.predict(Xtest))
			scored = output - np.array(ytest)

			#Counting as correct predictions the ones which fall in +/-1
			correct=0
			c = Counter(scored)
			for k in c.keys():
				if k<2 and k>-2:
					correct += c[k]
			
			score += float(correct)/len(scored)




		output = forest.predict(Xtest)
		#metricR = recall_score(ytest,output,average='micro')
		#metricP = precision_score(ytest,output,average='micro')

		#print('P / R: {0} , {1}  '.format(metricP,metricR))

		#score = cross_val_score(forest, Xtt, Y, cv=4, n_jobs=-1)
		#print('Scores with proper CV:')
		#print(score*100)

	
		#Averaging accuracy over folds
		print('Accuracy: {0} %    User: {1}'.format(score*100/folds,testUser))

		#totalP += metricP
		#totalR +=metricR
		acc += score*100/folds
		maxminAcc.append(score*100/folds)
		del Xlist[:]
		del ScreenList[:]
		#print('User: {0}  Accuracy: {1}'.format(testUser,tempAcc))
	print('Average accuracy: {0} % '.format(float(acc)/len(uids1)))
	print('Max / Min accuracy: {0}%  / {1}% '.format(max(maxminAcc), min(maxminAcc)))
	#print('Average precision: {0} %'.format(float(totalP)*100/len(uids1)))
	#print('Average recall: {0} %'.format(float(totalR)*100/len(uids1)))




if __name__ == '__main__':
	main()