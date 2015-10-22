import datetime,psycopg2
from collections import Counter
#converts unix timestamp to human readable date (e.g '1234567890' -> '2009-02-14  00:31:30')
def unixTimeConv(timestamp):
	newTime=str(datetime.datetime.fromtimestamp(int(timestamp)))
	year,time=newTime.split(' ')
	#z=x[2].split(':')
	return (year,time)

#counts occurence of each app for given user 'uid'
def countAppOccur(cur,uid):
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s ; """, [uid] )
	records = cur.fetchall()
	return Counter(records)



