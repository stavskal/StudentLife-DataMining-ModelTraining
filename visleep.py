import numpy as np
import pandas as pd
import json,csv,sys,os,psycopg2
from processingFunctions import *
from sleepNNreg import loadSleepLabels
from scipy.signal import medfilt

def movingaverage(values,window):
	weights = np.repeat(1.0,window)/window
	sma = np.convolve(values,weights,'valid')
	return sma



def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()





	if sys.argv[1]=='-vis':
		timestamp = 1368307369
		for u in uids2:
			for d in ['act','audio']:
				user= u +d
				cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(user, timestamp-18*3600, timestamp-9*3600) )
				rec = cur.fetchall()

				index = [item[0] for item in rec]
				data = [item[1] for item in rec]
				data = medfilt(data,11)
				s = pd.Series(data,index=index)
				ax = s.plot()
				fig = ax.get_figure()
				filen = 'raw'+ user +'.png'
				fig.savefig(filen)
				del s
				del fig
				del ax
				del data
				del index

		#do stuff



























if __name__ == '__main__':
	main(sys.argv[1:])