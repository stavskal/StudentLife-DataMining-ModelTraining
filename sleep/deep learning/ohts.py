
import datetime
import psycopg2
import numpy as np
import matplotlib.pyplot as pyp
import seaborn as sns
from six.moves import cPickle as pickle

uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u13','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']
uids16 = ['u49','u50','u51','u52','u53','u54','u56','u57','u58','u59']
def loadSleepLabels(cur,uid):
	uid = uid+'sleep'

	cur.execute('SELECT hour,time_stamp,rate FROM {0}'.format(uid))
	records = cur.fetchall()
	#records = sorted(records,key=lambda x:x[1])
	return(records)



class OneHotTimeSeries(object):
	""" Creates one hot encoded version of merged activity/audio
	timeseries in a 12 hour period around sleep quality responses
	"""

	def __init__(self,cur,user,response_ts,rate,hours):
		"""
		:cur:
			Cursor pointing to PostgreSQL database schema
		:user:
			User ID in the form of uXX, XX E[00,59]. Database tables assosiacted with
			each user use it in their name as well plus an acronym depicting the details.
			E.g. u00sleep table contains sleep information for user u00.

		:response_ts:
			Unix timestamp (GMT) for when participants reported their sleep quality

		:rate:
			Reported sleep quality (0/1 -> Good/Bad)

		:hours:
			Number of hours slept
		"""

		self.not_exists_au = 1
		self.not_exists_ac = 1
		self.not_big = 1
		self.rate = rate
		self.hours = hours
		self.cur = cur
		self.user = user
		self.response_ts = response_ts

		self.set_period()
		self.data_retrieval()
		if self.not_exists_au and self.not_exists_ac:
			self.align_timeseries()
			self.one_hot_encode()
		
	def set_period(self):
		"""
		Calculates two timestamps corresponding to 22:00 and 10:00 of previous
		day. This is the period from which data will be sampled to be fed
		into DNN architecture
		"""

		newTime = str(datetime.datetime.fromtimestamp(self.response_ts))
		#print(newTime)
		yearDate,timeT = newTime.split(' ')
		#year,month,day = str(yearDate).split('-')
		hour,minutes,sec = timeT.split(':')
		# Converting from UTC+2 to UTC-4
		hour = int(hour)-6
	#	print(int(minutes))
		self.am10 = self.response_ts - (int(hour)-10)*3600 - int(minutes)*60
		self.pm10 = self.response_ts - (int(hour)+2)*3600 - int(minutes)*60
		#print(self.am10,self.pm10)

	def data_retrieval(self):
		"""
		Retrieves activity/audio inference list from database between 22:00 and 10:00
		accompanied by their corresponding timestamps in the form of tuples
		Cell 0: timestamp
		Cell 1: activity/inference
		Then stores them as dictionary with timestamps as keys
		"""

		self.cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'
			.format(self.user+'act', self.pm10, self.am10) )
		records = self.cur.fetchall()
		if not records:
			print('No activity data found for user', self.user)
			self.not_exists_ac = 0
		# Accelerometer / audio classifier also classified as 'Unknown' (3rd class)
		# which does not provide any context, so discarding
		records = [item for item in records if item[1]!=3]
		self.activity_ts = {timestamp: inference for timestamp,inference in records}

		self.cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'
			.format(self.user+'audio', self.pm10, self.am10) )
		records = self.cur.fetchall()
		if not records:
			print('No audio data found for user', self.user)
			self.not_exists_au = 0

		records = [item for item in records if item[1]!=3]
		self.audio_ts = {timestamp: inference for timestamp,inference in records}


	def align_timeseries(self):
		"""
		Creates tuple time series with fixed length containing
		(activity,audio) data for every timestemp. Step in time is
		5 seconds atm, 86400 values in 12 hour period
		"""
	#	print(self.pm10,self.am10)
		self.tuple_ts = np.zeros([2,8640])
		current_time = self.pm10
		time_list_ac = np.array(self.activity_ts.keys())
		time_list_au = np.array(self.audio_ts.keys())
		
		distances_ac = np.zeros(8640)
		distances_au = np.zeros(8640)
		for i in range(0,8640):
			#Find position and value of nearest inference to time step
			diff_ac = np.abs(time_list_ac-current_time)
			diff_au = np.abs(time_list_au-current_time)

			nearest_ac_inference = np.argmin(diff_ac)
			nearest_au_inference = np.argmin(diff_au)

			# Keeping error for evaluation purposes
			distances_ac[i] = (time_list_ac[nearest_ac_inference]-current_time)
			distances_au[i] = (time_list_au[nearest_au_inference]-current_time)

			# Timelist[nearest] is the nearest timestamp, which is also the key
			# to get the inference from activity/audio_(ts)
			self.tuple_ts[0,i] = self.activity_ts[time_list_ac[nearest_ac_inference]]
			self.tuple_ts[1,i] = self.audio_ts[time_list_au[nearest_au_inference]]
			current_time += 5
		#print(self.user)
		#print(np.mean(distances_ac), np.median(distances_ac))
		#print(np.mean(distances_au), np.median(distances_au))

		if np.abs(np.mean(distances_ac))>1.5 or np.abs(np.mean(distances_au))>1.5:
			self.not_big = 0

	#	ax = sns.kdeplot(distances_ac, shade=True)
		#fig = ax.get_figure()
	#	fig.savefig(self.user)
	#	exit()
		print('---------------------------------')
		#print(self.tuple_ts[:,100])
		#print(self.tuple_ts[:,200])

	def one_hot_encode(self):
		"""
		Converts tuple timeseries to one hot encoding mapping
		all possible N combinations of audio/activity classes
		to 1-of-N vector  
		"""
		self.oht = np.zeros([9,8640])
		for i in range(0,8640):
			if self.tuple_ts[0,i]==0 and self.tuple_ts[1,i]==0:
				self.oht[0,i] = 1
			elif self.tuple_ts[0,i]==0 and self.tuple_ts[1,i]==1:
				self.oht[1,i] = 1
			elif self.tuple_ts[0,i]==0 and self.tuple_ts[1,i]==2:
				self.oht[2,i] = 1
			elif self.tuple_ts[0,i]==1 and self.tuple_ts[1,i]==0:
				self.oht[3,i] = 1
			elif self.tuple_ts[0,i]==1 and self.tuple_ts[1,i]==1:
				self.oht[4,i] = 1
			elif self.tuple_ts[0,i]==1 and self.tuple_ts[1,i]==2:
				self.oht[5,i] = 1
			elif self.tuple_ts[0,i]==2 and self.tuple_ts[1,i]==0:
				self.oht[6,i] = 1
			elif self.tuple_ts[0,i]==2 and self.tuple_ts[1,i]==1:
				self.oht[7,i] = 1
			elif self.tuple_ts[0,i]==2 and self.tuple_ts[1,i]==2:
				self.oht[8,i] = 1

	#	print(self.oht[:,100])
		#print(self.oht[:,200])
		#print(self.oht.shape)



try:
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()

except psycopg2.DatabaseError as err:
	print('Error %s' % err)
	exit()
uids1=['u02']
all_rates =[]
all_ts = []
for u in uids16:
	sleep_labels = loadSleepLabels(cur,u)
	for index,val in enumerate(sleep_labels):
		a = OneHotTimeSeries(cur=cur, user=u, response_ts=val[1], rate=val[2], hours=val[0])
		if a.not_exists_ac and a.not_exists_au and a.not_big:
			all_ts.append(a.oht)
			all_rates.append(a.rate)
print(len(all_ts))
arr = np.array(all_ts, dtype='int16')

#np.save('ohts.npy',arr)
y = np.array(all_rates)
#np.save('rates.npy',y)

try:
	f = open('ohts59.pickle', 'wb')
	save = {
	'X': arr,
	'y': y,
	}
	pickle.dump(save,f, pickle.HIGHEST_PROTOCOL)
	f.close()
except:
	print('giati gamiesai twra')
	raise

