import json,csv,sys,os,psycopg2
import numpy as np
from collections import Counter 
from processingFunctions import *
#from sklearn.neural_network import MLPRegressor 
import matplotlib.pyplot as plt
import time


def loadSleepLabels(cur,uid):
	uid = uid+'sleep'

	cur.execute('SELECT hour,time_stamp FROM {0}'.format(uid))
	records = cur.fetchall()

	#hours = [item[0] for item in records]
	#times = [item[1] for item in records]

	return records 


# returns duration screen remained locked during previous evening and night
# used for Sleep Estimator as feature
def screenLockDur(cur,uid,timestamp):
	totalDur = 0
	uidL = uid + 'lock'
	cur.execute('SELECT * FROM {0} WHERE timeStart>={1} AND timeStop <= {2}'.format(uidL, timestamp-86400, timestamp) )
	records = cur.fetchall()
	#keeping only records during evening epoch ( 18:00 < t < 9:00)
	tStart = [item[0] for item in records]
	timeEpochs = epochCalc(tStart)

	
	for i in range(0,len(tStart)):
		if timeEpochs[i][0] =='night':
			totalDur += records[i][1] -records[1][0]

	return(totalDur)



#def stationaryDUr(cur,uid,timestamp):






def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()





	if sys.argv[1]=='-train':
		sld = screenLockDur(cur,'u00',1365396215)
		#reg = MLPRegressor(hidden_layer_sizes=2 ,activation='logistic', algorithm='l-bfgs')
		#do stuff
		




























if __name__ == '__main__':
	main(sys.argv[1:])