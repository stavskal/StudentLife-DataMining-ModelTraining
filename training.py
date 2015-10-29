import json,csv,sys,os,psycopg2
from processingFunctions import  computeAppStats, countAppOccur, appTimeIntervals



day = 86400



def computeAppStats(cur,uid,timestamp):
	appOccurTotal = countAppOccur(cur,uid)
	appStats=[]

	i=0
	
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