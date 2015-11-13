import json,csv,sys,os,psycopg2
from netaddr import EUI
from processingFunctions import uids

create = "CREATE TABLE {0} (time_stamp INT, mac VARCHAR(100), class_id INT, level INT) ;"
insert = "INSERT INTO {0} (time_stamp, mac, class_id,level) VALUES ({1},{2},{3},{4}) ;"
drop = "DROP TABLE {0}"

create1= """CREATE TABLE {0} (start_timestamp INT, end_timestamp INT) ; """
insert1="INSERT INTO {0} (start_timestamp,end_timestamp	) VALUES ({1},{2})	;"

create2= """CREATE TABLE {0} (time_stamp INT, activity INT) ; """
insert2="INSERT INTO {0} (time_stamp, activity) VALUES ({1},{2})	;"

# IMPROTANT NOTE
# MAC addresses are stored in their oct() form as strings



# Parses data from CSV files to insert bluetooth scans in database tables
def dbInsertBTscan(csvfile,cur,tableName):
	uid=(csvfile.split('_'))[1][0:3] +tableName
	cur.execute(create.format(uid))
	print(create.format(uid))
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				#print(uid)
				#data=[str(uid), record['time'],record['MAC'],record['class_id'],record['level']]
				a = EUI(record['MAC'])
				#data=(uid,record['time'],str(record['MAC']),record['class_id'],record['level'],)
				insertQ = insert.format(uid,record['time'],oct(a),record['class_id'],record['level'])
				cur.execute(insertQ)


# Parses data from CSV files to insert bluetooth scans in database tables
def dbInsertCon(csvfile,cur,tableName):
	uid=(csvfile.split('_'))[1][0:3] +tableName
	cur.execute(create1.format(uid))
	print(create1.format(uid))
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				
				insertQ = insert1.format(uid,record['start_timestamp'], record[' end_timestamp'])
				cur.execute(insertQ)


def dbInsertAct(csvfile,cur,tableName):
	uid=(csvfile.split('_'))[1][0:3] +tableName
	cur.execute(create2.format(uid))
	print(create2.format(uid))
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				insertQ = insert2.format(uid,record['timestamp'], record[' activity inference'])
				cur.execute(insertQ)


def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()



	# if user choses '-insert' then bluetooth and conversation data will be loaded
	if sys.argv[1]=='-insert':
		


		#setting directory to load app_usage information
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/bluetooth'
		tableName = 'bt'
		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			filename = directory +'/'+ filename

			dbInsertBTscan(filename,cur,tableName)

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/conversation'
		tableName = 'con'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			filename = directory +'/'+ filename

			dbInsertCon(filename,cur,tableName)

		
	# if user choses '-insert1' then activity data will be loaded
	elif sys.argv[1]=='-insert1':
		#setting directory to load app_usage information
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/activity'
		tableName = 'act'
		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			filename = directory +'/'+ filename

			dbInsertAct(filename,cur,tableName)




	elif sys.argv[1]=='-drop':

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/bluetooth'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]+'bt'
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/conversation'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]+'con'
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/activity'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]+'act'
			dropQ=drop.format(uid)
			cur.execute(dropQ)



	con.commit()
	con.close()



if __name__ == '__main__':
	main(sys.argv[1:])