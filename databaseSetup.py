import csv,codecs,os
from pymongo import MongoClient


client = MongoClient()
db = client.dataset


def dbInsertData(csvfile,c):	
	with open(csvfile,'rb') as inCsv:
		parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
		for record in parsed:

			#inserting data in to proper collection inside DB
			if c == 'app_usage':
				db.app_usage.insert_one(record)

			elif c == 'sms':
				db.sms.insert_one(record)

			elif c == 'call_log':
				db.call_log.insert_one(record)

			elif c == 'calendar':
				db.calendar.insert_one(record)

			





def main():

	#Below four folders containing app_usage,sms,call_logs,calendar are being
	#loaded and inserted RAW in the database for further processing


	#setting directory to load app_usage information
	directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/app_usage'

	#inserting all files to database
	for filename in os.listdir(directory):
		filename= directory +'/'+ filename
		dbInsertData(filename,'app_usage')

	print('Done with app_usage')
	#setting directory to load sms information
	directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sms'

	#inserting all files to database
	for filename in os.listdir(directory):
		filename= directory +'/'+ filename
		dbInsertData(filename,'sms')


	print('Done with sms')

	#setting directory to load sms information
	directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/call_log'

	#inserting all files to database
	for filename in os.listdir(directory):
		filename= directory +'/'+ filename
		dbInsertData(filename,'call_log')

	print('Done with call_logs')
	#setting directory to load sms information
	directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/calendar'

	#inserting all files to database
	for filename in os.listdir(directory):
		filename= directory +'/'+ filename
		dbInsertData(filename,'calendar')

	print('Done with calendar')





if __name__ == '__main__':
	main()