
import datetime
import psycopg2

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

	def set_period(self):
		"""
		Calculates two timestamps corresponding to 22:00 and 10:00 of previous
		day. This is the period from which data will be sampled to be fed
		into DNN architecture
		"""

		newTime = str(datetime.datetime.fromtimestamp(self.response_ts))
		print(newTime)
		yearDate,timeT = newTime.split(' ')
		year,month,day = str(yearDate).split('-')
		hour,minutes,sec = timeT.split(':')
		# Converting from UTC+2 to UTC-4
		hour = int(hour)-6
		print(int(minutes))
		self.am10 = self.response_ts - (int(hour)-10)*3600 - int(minutes)*60
		self.pm10 = self.response_ts - (int(hour)+2)*3600 - int(minutes)*60
		print(self.am10,self.pm10)

	def data_retrieval(self):
		self.cur.execute('SELECT * FROM {0} WHERE time_stamp>={1} AND time_stamp<={2}'
			.format(self.user+'act', self.pm10, self.am10) )
		records = self.cur.fetchall()
		print(len(records))



	#def time_interval12(self):
try:
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()

except psycopg2.DatabaseError as err:
	print('Error %s' % err)
	exit()

a = OneHotTimeSeries(cur=cur, user='u00', response_ts=1366488754, rate=1)
