import datetime,psycopg2
from collections import Counter

#---------------------------------------------
#This script contains a collection of functions
#that are useful in processing the data in
#StudentLife Dataset
#---------------------------------------------

hour = 3600
day = 86400
halfday = 43200
weekSec = 604000

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

uids1=['u00','u24','u08','u57','u52','u51','u36']


#---------------------------------------------------------------------------------------
# converts unix timestamp to human readable date (e.g '1234567890' -> '2009 02 14  00 31 30')
def unixTimeConv(timestamp):
	newTime = str(datetime.datetime.fromtimestamp(int(timestamp)))
	yearDate,timeT = newTime.split(' ')
	year,month,day = yearDate.split('-')
	hour,minutes,sec = timeT.split(':')
	return (year,month,day,hour,minutes,sec)

#----------------------------------------------------------------------------------------
# converts timestamp to epoch (day,evening,night) pretty straightforward right?
def epochCalc(timestamp):
	year,month,day,hour,minutes,sec = unixTimeConv(timestamp)
	hour=int(hour)
	if hour >= 9 and hour <=18:
		epoch='day'
	elif hour>0 and hour <9:
		epoch='night'
	else:
		epoch='evening'

	return epoch

#-----------------------------------------------------------------------
# computes average time of stress reports in order to find the optimal 
# time window for the app statistics calculation to reduce overlapping of features
def meanStress(cur,uid):
	records = sorted( loadStressLabels(cur,uid) , key=lambda x:x[0] )
	mean = 0 

	for i in range(0,len(records)-1):
		mean += records[i+1][0] - records[i][0]

	mean = float(mean) / len(records)
	return(mean)


#---------------------------------------------------------------------------------------
# counts occurences of 'mc' most common apps for given user 'uid' during experiment
def countAppOccur(cur,uid,mc,timeQuery):
	meanS = meanStress(cur,uid)
	cur.execute("SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp <= %s AND time_stamp >= %s;",[uid,timeQuery,timeQuery-meanS])

	#Counter class counts occurrences of unique ids, and the 50 most common are kept
	records =Counter( cur.fetchall() )
	
	#transforming keys cause of ugly return shape of Counter class
	for k in records.keys():
		records[k[0]] = records.pop(k)

	return records


#---------------------------------------------------------------------------------------
# computes application usage frequency and number of unique apps per uid for given
def computeAppStats(cur,uid,timeWin):
	appOccur = countAppOccur(cur,uid)
	appStats=[]

	#selecting start and end date of data acquisition
	cur.execute("""SELECT min(time_stamp),max(time_stamp) FROM appusage WHERE uid = %s""",[uid])
	records = cur.fetchall()

	#tStart: timestamp logging started, tEnd: timestamp loggind ended
	tStart , tEnd = records[0][0], records[0][1]
	durationTotal = tEnd - tStart


	i=0
	# iterating until we reach the end of sampling period. In every iteration only the data contained in
	# given time window are selected from database.
	# statistic computed is the daily usage frequency(if today used 10 times, 100 total then 0.1 (10%) is returned)
	# a list of dictionaries is returned, each cell containing a dictionary (key:task_id, value: freq) for a time period 
	while(tStart+timeWin<tEnd):
		tAfter = tStart + timeWin
		cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp > %s AND time_stamp < %s ; """, [uid,tStart,tAfter] )
		records= Counter( cur.fetchall() )

		#transforming keys to be in readable format
		for k in records.keys():
			records[k[0]] = records.pop(k)

		# number of unique applications 
		uniqueApps = len(records.keys())
		
		# usageFrequency:  number of times in timeWin / total times
		usageFrequency= {k: float(records[k])*100/float(appOccur[k]) for k in appOccur.viewkeys() & records.viewkeys() }
		

		appStats.append(usageFrequency)
		
		tStart = tStart + timeWin
		tAfter = tAfter + timeWin

		i=i+1

	return appStats
#---------------------------------------------------------------------------------------


		

#---------------------------------------------------------------------------------------
# calculates time between subsequent application usages
# IMPORTANT: many applications are running in the background during all sampling times (every 1200sec)
# They have not been included in returned list, they were cross-checked with phone screen data
def appTimeIntervals(cur,uid,timestamp,timeWin):
	timeInterval=[]
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s ; """, [uid] )

	#allKeys holds all the running_task_ids as keys of dict
	allKeys = dict(Counter( cur.fetchall())).keys() 
	#unique holds the total number of unique applications
	unique = len(allKeys)

	cur.execute("""SELECT running_task_id,time_stamp  FROM appusage WHERE uid = %s AND time_stamp>=%s AND time_stamp <=%s  ; """, [uid,timestamp-timeWin,timestamp] )
	records = cur.fetchall()	
	

	for k in range(0,unique):
		#singleAppOccur holds all usages+time of one application (each iter) for given user
		singleAppOccur = [item for item in records if allKeys[k][0] in item]
		#sorting app usages according to time in order to compute intervals between subsequent uses
		sortedTimestamp = sorted(singleAppOccur, key=lambda x:x[1] )

		timeInterval.append([])
		for use in range(0,len(sortedTimestamp)-1):

			#checking if screen was on during appusage to only count user interactions, not background processes
			if checkScreenOn(cur, uid, sortedTimestamp[use][1]) == True and checkScreenOn(cur, uid, sortedTimestamp[use+1][1]):
				timeInterval[k].append(sortedTimestamp[use+1][1] - sortedTimestamp[use][1]) 
		
	#timeIntervals is a list of lists, each row contains consequent uses of signle application (also a list)
	return timeInterval

#---------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------
# returns True if screen was On at given time, false otherwise
# used to determine whether application was user initiated or not
def checkScreenOn(cur,uid,time):
	uid = uid +'dark'
	time = int(time)
	cur.execute("SELECT * FROM {0} WHERE timeStart <= {1} AND timeStop >={2} ; ".format(uid,time,time) )
	records = cur.fetchall()

	if not records:
		return(False)
	else:
		return(True)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# retrieves stress labels for user 'uid' and returns a list with their corresponding timestamps
def loadStressLabels(cur,uid):
	cur.execute("SELECT time_stamp,stress_level  FROM {0} ".format(uid) )
	records = cur.fetchall()

	return records	






#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
t = 1366007398
d = countAppOccur(cur,'u59',30,t)
print(d)
print('--------------------------------------------------------')
for i in range(0,10):
	t += day
	d = d + countAppOccur(cur,'u59',30,t)
	print(d)

	print('-----------------------------------------------------')

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

#TODO: visualize stuff to gain more insight
#TODO: train model on data (?)s


#TODO for thursday meeting: short term timeplan