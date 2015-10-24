import datetime,psycopg2
from collections import Counter

#---------------------------------------------
#This script contains a collection of functions
#that are useful in processing the data in
#StudentLife Dataset
#---------------------------------------------

hour = 3600
day = 86400
weekSec = 604000

#---------------------------------------------------------------------------------------
#converts unix timestamp to human readable date (e.g '1234567890' -> '2009 02 14  00 31 30')
def unixTimeConv(timestamp):
	newTime = str(datetime.datetime.fromtimestamp(int(timestamp)))
	yearDate,timeT = newTime.split(' ')
	year,month,day = yearDate.split('-')
	hour,minutes,sec = timeT.split(':')
	return (year,month,day,hour,minutes,sec)


#---------------------------------------------------------------------------------------
#counts occurence of each app for given user 'uid' during experiment
def countAppOccur(cur,uid):
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s ; """, [uid] )

	#Counter class counts occurrences of unique ids, and the 200 most common are kept
	records =dict(Counter( cur.fetchall() ).most_common(200))
	
	return records


#---------------------------------------------------------------------------------------
#computes application usage frequency and number of unique apps per uid per timeWin (hour, day, week)
def computeAppStats(cur,uid,timeWin):
	appOccur = countAppOccur(cur,uid)

	#selecting start and end date of data acquisition
	cur.execute("""SELECT min(time_stamp),max(time_stamp) FROM appusage WHERE uid = %s""",[uid])
	records = cur.fetchall()

	#tStart: timestamp logging started, tEnd: timestamp loggind ended
	tStart , tEnd = records[0][0], records[0][1]
	durationTotal = tEnd - tStart


	i=0
	while(tStart+timeWin<tEnd):
		tAfter = tStart + timeWin
		cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s AND time_stamp > %s AND time_stamp < %s ; """, [uid,tStart,tAfter] )
		print('----------------------- Epoch {0}---------------------------------'.format(i))
		records= Counter( cur.fetchall() )

		#number of every days' unique applications 
		dailyUniqueApps = len(records.keys())
		
		# dailyUsageFrequency:  number of times today / total times
		dailyUsageFrequency= {k: float(records[k])*100/float(appOccur[k]) for k in appOccur.viewkeys() & records.viewkeys() }
		print(dailyUsageFrequency)
		print('--------------------------------------------------------')

		tStart = tStart + timeWin
		tAfter = tAfter + timeWin

		i=i+1
		

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
occurences=[]
def appTimeIntervals(cur,uid):
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s ; """, [uid] )

	#unique holds the total number of unique applications
	allKeys = dict(Counter( cur.fetchall())).keys() 
	unique = len(allKeys)

	cur.execute("""SELECT running_task_id,time_stamp  FROM appusage WHERE uid = %s ; """, [uid] )

	records = cur.fetchall()

	print(allKeys[1][0])
	
	#not tested yet
	for k in range(1,unique):
		occurences[k].append( [item for item in records if allKeys[k][0] in item])
		print(occurences[k])
		



	


#testing
con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
#computeAppStats(cur,'u00',day)
appTimeIntervals(cur,'u00')
#countAppOccur(cur,'u01')



#[DONE]: function that produces labels [stressed/not stressed] from surveys [DONE]
#[DONE]: function that computes application usage statistics in time window (day/week) (frequency, mean, dev)

#TODO: function that computes time intervals between consequent application usages
#TODO: function that computes sms+calls statistical features in time window (how many sms, how many people)
#NOTE: some call+sms logs do not contain any data (maybe corrupted download?)

#TODO: visualize stuff to gain more insight
#TODO: train model on data (?)s


#TODO for thursday meeting: project proposal on which direction I want this to move (stress background, state of art, what i reviewd in literature) (abstract kind of)
#TODO for thursday meeting: short term timeplan