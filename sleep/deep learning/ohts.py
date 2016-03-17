
import datetime
import psycopg2
import numpy as np


class OneHotTimeSeries(object):
	""" Creates one hot encoded version of merged activity/audio
	timeseries in a 12 hour period around sleep quality responses
	"""

	def __init__(self,cur,user,response_ts,rate):
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
		"""
		self.cur = cur
		self.user = user
		self.response_ts = response_ts
		self.set_period()
		self.data_retrieval()
		self.align_timeseries()
		self.one_hot_encode()

	def set_period(self):
		"""
		Calculates two timestamps corresponding to 22:00 and 10:00 of previous
		day. This is the period from which data will be sampled to be fed
		into DNN architecture
		"""

		newTime = str(datetime.datetime.fromtimestamp(self.response_ts))
		print(newTime)
		yearDate,timeT = newTime.split(' ')
		#year,month,day = str(yearDate).split('-')
		hour,minutes,sec = timeT.split(':')
		# Converting from UTC+2 to UTC-4
		hour = int(hour)-6
	#	print(int(minutes))
		self.am10 = self.response_ts - (int(hour)-10)*3600 - int(minutes)*60
		self.pm10 = self.response_ts - (int(hour)+2)*3600 - int(minutes)*60
		print(self.am10,self.pm10)

	def data_retrieval(self):
		"""
		Retrieves activity/audio inference list from local database between 22:00 and 10:00
		accompanied by thei corresponding timestamps in the form of tuples
		Cell 0: timestamp
		Cell 1: activity/inference
		Then stores them as dictionary with timestamps as keys
		"""

		self.cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'
			.format(self.user+'act', self.pm10, self.am10) )
		records = self.cur.fetchall()
		if not records:
			print('No activity data found')
			exit()
		# Accelerometer / audio classifier also classified as 'Unknown' (3rd class)
		# which does not provide any context, so discarding
		records = [item for item in records if item[1]!=3]
		self.activity_ts = {timestamp: inference for timestamp,inference in records}

		self.cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'
			.format(self.user+'audio', self.pm10, self.am10) )
		records = self.cur.fetchall()
		if not records:
			print('No audio data found')
			exit()
		records = [item for item in records if item[1]!=3]
		self.audio_ts = {timestamp: inference for timestamp,inference in records}


	def align_timeseries(self):
		"""
		Creates tuple time series with fixed length containing
		(activity,audio) data for every timestemp. Step in time is
		5 seconds atm, 86400 values in 12 hour period
		"""
		print(self.pm10,self.am10)
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
		
		print(np.mean(distances_ac), np.median(distances_ac))
		print(np.mean(distances_au), np.median(distances_au))
		print(self.tuple_ts[:,100])
		print(self.tuple_ts[:,200])

	def one_hot_encode(self):
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

		print(self.oht[:,100])
		print(self.oht[:,200])
		print(self.oht.shape)


	#def time_interval12(self):
try:
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()

except psycopg2.DatabaseError as err:
	print('Error %s' % err)
	exit()

a = OneHotTimeSeries(cur=cur, user='u00', response_ts=1364580795, rate=1)
print(a.am10-a.pm10)