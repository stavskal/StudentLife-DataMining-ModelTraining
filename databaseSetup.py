import csv,codecs
from pymongo import MongoClient


global_mongo = None
global_db = None
global_coll = None

client = MongoClient()
db = client.test1


def dbSetup():	
	i=0
	csvfile='running_app_u00.csv'
	with open(csvfile,'rb') as inCsv:
		parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
		for record in parsed:
			string_record=dict([(str(k), v) for k, v in record.items()])
			add_record_to_mongo(record,i)
			i= i+1
			



#code from github/manchicken
def add_record_to_mongo(record,i):
	
	print i
	# Now let's insert
	db.test1.insert_one(record)





def main():
	dbSetup()


if __name__ == '__main__':
	main()