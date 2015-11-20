import numpy as np
import pandas as pd
import matplotlib.pyplot as pyp
import json,csv,sys,os,psycopg2
from collections import Counter
from processingFunctions import *
from sleepNNreg import loadSleepLabels
from scipy.signal import medfilt


# Pipeling for defining sleep time
# Load to pandas time series 


def most_common(lst):
	data = Counter (list(lst))
	#print(data,data[0],data[1],data[2])
	if data[0] > data[1]+data[2]:
		return 1
	else:
		return 0
	#return data.most_common(1)[0][0]


def sleep(ser1,ser2):
	seri= []
	index = []
	winLen=10
	j = 0
	for i in range(0,len(ser1)-winLen,winLen):
		tempWindow = np.concatenate( (np.array(ser1[i:i+winLen]) , np.array(ser2[i:i+winLen])),axis=0)
		#print(tempWindow)
		if most_common(tempWindow)==0:
			seri.append(0)
		else:
			seri.append(1)
		j += 1
		index.append(j)

	s = pd.Series(seri,index=index)
	pyp.figure()
	ax = s.plot()
	fig = ax.get_figure()
	filen = 'testaaaa.png'
	fig.savefig(filen)




def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()



	if sys.argv[1]=='-vis':
		s =[]
		timestamp = 1367757820	
		for u in uids2:
			i=0
			for d in ['act', 'audio']:
				user= u +d
				cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'.format(user, timestamp-14*3600, timestamp-7*3600) )
				rec = cur.fetchall()

				index = [item[0] for item in rec if item[1]!=3]
				print(len(index))
				data = [item[1] for item in rec if item[1]!=3]
				data = pd.Series(medfilt(data,13),index=index)
				s.append(data)
				#ax = pd.Series(data,index=index).plot()
				#fig = ax.get_figure()
				#filen = 'raw'+ user +'.png'
				#fig.savefig(filen)
			#	del ax
			#sleep(s[0],s[1])
			con  = pd.concat((s[0],s[1]))
			print(Counter(con))
			print(con[100:101].values)
			for i in range(0,len(con)):
				if con[i:i+1].values==0:
					print(con[i:i+1].values)
			
			ax= con.plot()
			fig = ax.get_figure()
			fig.savefig('giapame.png')
		#do stuff



























if __name__ == '__main__':
	main(sys.argv[1:])