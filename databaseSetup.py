import csv,codecs,os
from pymongo import MongoClient


global_mongo = None
global_db = None
global_coll = None

client = MongoClient()
db = client.AppUsage


def dbSetup(csvfile):	
	i=0
	with open(csvfile,'rb') as inCsv:
		parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
		for record in parsed:
			#string_record=dict([(str(k), v) for k, v in record.items()])
			add_record_to_mongo(record,i)
			i= i+1
			



#code from github/manchicken
def add_record_to_mongo(record,i):
	
	print i
	# Now let's insert
	db.logs.insert_one(record)





def main():

	#current directory script being run
	directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/app_usage'
	
	for filename in os.listdir(directory):
		filename= directory +'/'+ filename

		dbSetup(filename)


if __name__ == '__main__':
	main()