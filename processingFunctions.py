import datetime,psycopg2
from collections import Counter

#---------------------------------------------
#This script contains a collection of functions
#that are useful in processing the data in
#StudentLife Dataset
#---------------------------------------------


#converts unix timestamp to human readable date (e.g '1234567890' -> '2009-02-14  00:31:30')
def unixTimeConv(timestamp):
	newTime=str(datetime.datetime.fromtimestamp(int(timestamp)))
	year,time=newTime.split(' ')
	return (year,time)

#counts occurence of each app for given user 'uid' during experiment
def countAppOccur(cur,uid):
	cur.execute("""SELECT running_task_id  FROM appusage WHERE uid = %s ; """, [uid] )
	records = cur.fetchall()
	print Counter(records)





#[DONE]: function that produces labels [stressed/not stressed] from surveys [DONE]


#TODO: function that computes application usage statistics in time window (day/week) (frequency, mean, dev)

#TODO: function that computes sms+calls statistical features in time window (how many sms, how many people)
#NOTE: some logs do not contain any data (maybe corrupted download?)

#TODO: visualize stuff to gain more insight
#TODO: train model on data (?)


#TODO for thursday meeting: project proposal on which direction I want this to move (stress background, state of art, what i reviewd in literature) (abstract kind of)
#TODO for thursday meeting: short term timeplan