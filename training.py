import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from processingFunctions import  computeAppStats, countAppOccur, appTimeIntervals
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


day = 86400
halfday = 43200
quarterday = 21600

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

uids1=['u10','u16','u19','u32','u33','u43','u44','u49','u57','u59']





# returns feature vector corresponing to (timestamp,stress_level) (report)
# This feature vector is of size len(uniqueApps), total number of different apps for user
# each cell corresponds to the % of time for one app. Apps that were not used during 
# previous day simply have zero in feature vector cell (sparse)
def appStatsL(cur,uid,timestamp):
	appOccurTotal = countAppOccur(cur,uid)
	keys = np.fromiter(iter(appOccurTotal.keys()), dtype=int)
	keys = np.sort(keys)
	appStats1 = np.zeros(keys[-1])
	appStats=[]
	
	tStart = timestamp - halfday

	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp > %s AND time_stamp < %s ; """, [uid,tStart,timestamp] )
	records= Counter( cur.fetchall() )
	for k in records.keys():
		records[k[0]] = records.pop(k)


	for i in keys:
		if i in records.keys():
			appStats1[i-1] = float(records[i])*100 / float(appOccurTotal[i])

	



	# number of unique applications 
	#uniqueApps = len(records.keys())
	# usageFrequency:  number of times in timeWin / total times
	#usageFrequency= {k: float(records[k])*100/float(appOccurTotal[k]) for k in appOccurTotal.viewkeys() & records.viewkeys() }
	#appStats.append(usageFrequency)

	return appStats1




#---------------------------------------------------------------------
# computes the total time (sec) that screen was on during the past day
def timeScreenOn(cur,uid,timestamp):
	#table name is in form: uXXdark
	uid = uid +'dark'
	#tStart is exactly 24h before given timestamp
	tStart = timestamp - day

	#fetching all records that fall within this time period
	cur.execute('SELECT timeStart,timeStop FROM {0} WHERE timeStart >= {1} AND timeStop <= {2}'.format(uid,tStart,timestamp))
	records = cur.fetchall()

	totalTime =0
	# each tuple contains the time period screen was on. Calculate its duration and add it to total
	for k in records:
		totalTime += k[1]-k[0]

	return totalTime





#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
acc=0


# ------------TEST CASE-----------------------------
# 10 user were picked from the dataset
# 70% of their stress reports and the corresponding 24h features are used for training
# the rest 30% is used for testing. The train/test reports are randomly distributed
# throughout the whole experiment duration. No FV is used both for training and testing
# after the 10 models are trained and tested, the overall accuracy is averaged
# Random Forests were picked due to their 'universal applicability', each with 20 decision trees

for testUser in uids1:
	#testUser='u49'
	print(testUser)

	cur.execute("SELECT time_stamp,stress_level FROM {0}".format(testUser))

	records = cur.fetchall()
	#print(len(records))

	# The intended thing to achieve here is to calculate the feature vector(FV) in the 24h period proceeding each 
	# stress report. Xtrain's rows are those FVs for ALL stress report timestamps 
	# DONE: change datatype from list to more efficient, e.g numpy 2D array
	a=appStatsL(cur,testUser,records[0][0])

	trainLength= int(0.7 * (len(records)))
	Xtrain = np.empty([trainLength, len(a)], dtype=float)
	Ytrain = np.empty([trainLength],dtype=int)

	testLength= int(0.3 *len(records))
	Xtest = np.empty([testLength, len(a)], dtype=float)
	Ytest = np.empty(testLength,dtype=int)


	used=[]
	for i in range(0,trainLength):
		trainU = random.choice(records)
		while trainU in used:
			trainU = random.choice(records)
		used.append(trainU)
		Xtrain[i] = appStatsL(cur,testUser,trainU[0])
		Ytrain[i] = trainU[1]

	for i in range (0,testLength):
		testU = random.choice(records)
		while testU in used:
			testU = random.choice(records)
		used.append(testU)
		Xtest[i] = appStatsL(cur,testUser,testU[0])
		Ytest[i] = testU[1]

	#i=0
	#for testU in records:
	#	Xtest[i] = appStatsL(cur,testUser,testU[0])
	#	Ytest[i] = testU[1]
	#	i = i+1

	#print(Ytest)

	forest = RandomForestClassifier(n_estimators=25,n_jobs=4)
	forest = forest.fit(Xtrain,Ytrain)

	output = forest.predict(Xtest)
	
	metricP = precision_score(Ytest,output, average='weighted')
	metricR = recall_score(Ytest,output, average='weighted')

	tempAcc = forest.score(Xtest,Ytest)
	acc += tempAcc
	print('P,R: {0} , {1} '.format(metricP,metricR))
	print('Fscore: {0}'.format ( 2*metricP*metricR/(metricR+metricP)))
	print('Accuracy: {0} %'.format(tempAcc*100))

print('Average accuracy: {0} %'.format(float(acc)*100/len(uids1)))
"""


def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()





	if sys.argv[1]=='-insert':


		#do stuff
		

	elif sys.argv[1]=='-drop':
		#do stuff


























if __name__ == '__main__':
	main()

"""
	
