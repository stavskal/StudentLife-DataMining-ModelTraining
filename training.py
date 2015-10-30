import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
from processingFunctions import  computeAppStats, countAppOccur, appTimeIntervals
from sklearn.ensemble import RandomForestClassifier


day = 86400
uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']






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
	
	tStart = timestamp - 2*day

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
testUser='u43'

cur.execute("SELECT time_stamp,stress_level FROM {0}".format(testUser))

records = cur.fetchall()
print(len(records))

# The intended thing to achieve here is to calculate the feature vector(FV) in the 24h period proceeding each 
# stress report. Xtrain's rows are those FVs for ALL stress report timestamps 
# DONE: change datatype from list to more efficient, e.g numpy 2D array
a=appStatsL(cur,testUser,records[0][0])
trainLength= int(0.7 * (len(records)))
Xtrain = np.empty([trainLength, len(a)], dtype=float)
Ytrain = np.empty([trainLength],dtype=int)

testLength= int(0.25 *len(records))
Xtest = np.empty([testLength, len(a)], dtype=float)
Ytest = np.empty(testLength,dtype=int)

i=0
for s in records:

	if i==trainLength+testLength:
		break

	if i>=trainLength and i < trainLength+testLength:
		Xtest[i-trainLength] = appStatsL(cur,testUser,s[0])
		Ytest[i-trainLength] = s[1]
		
	else:
		Xtrain[i] = appStatsL(cur,testUser,s[0])
		Ytrain[i] = s[1]

	
		
	i += 1

print(Ytest)

forest = RandomForestClassifier(n_estimators=10)

forest = forest.fit(Xtrain,Ytrain)

output = forest.predict(Xtest)
print(output)

acc = forest.score(Xtest,Ytest)
print(acc)
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
	
