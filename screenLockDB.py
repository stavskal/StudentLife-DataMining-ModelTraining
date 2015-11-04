import json,csv,sys,os,psycopg2


create1= """CREATE TABLE {0} (timeStart INT, timeStop INT) ; """
insert1="INSERT INTO {0} (timeStart,timeStop) VALUES ({1},{2})	;"
drop = "DROP TABLE {0};"

#-----------------------------------------------------------------
# This script inserts phonelock and phone screen on data into tables
# one table for each user. Phone screen ON/OFF data are in tables: 'uXXdark'
# Phone lock data are in tables: 'uXXlock'


def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()
	
	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()


	if sys.argv[1]=='-insert':

		#-----------------!!!!!!!!!!!!!-------------------------
		# PHONE SCREEN ON/OFF DATA
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/dark/'

		for filename in os.listdir(directory):

			#creating table for each user with name: 'uXX'
			uid = (filename.split('_'))[1][0:3]+ 'dark'
			createQ = create1.format(uid)
			cur.execute(createQ)
			

			#opening each csv file and extracting information
			with open(os.path.join(directory,filename),'rb') as inCsv:
				parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')

				#inserting records to DB
				for record in parsed:
					insertQ = insert1.format(uid, record['start'], record['end'])
					cur.execute(insertQ)


		# ------------!!!!!!!!!!!!-----------------------------
		# PHONE LOCK DATA
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/phonelock/'

		for filename in os.listdir(directory):

			#creating table for each user with name: 'uXX'
			uid = (filename.split('_'))[1][0:3]+ 'lock'
			createQ = create1.format(uid)
			cur.execute(createQ)
			

			#opening each csv file and extracting information
			with open(os.path.join(directory,filename),'rb') as inCsv:
				parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')

				#inserting records to DB
				for record in parsed:
					insertQ = insert1.format(uid, record['start'], record['end'])
					print(insertQ)
					cur.execute(insertQ)

		con.commit()
		con.close()

		#do stuff
	

	# -drop argument destroys all tables with phonelock and screen on/off data
	elif sys.argv[1]=='-drop':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/dark'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3] +'dark'
			dropQ=drop.format(uid)
			cur.execute(dropQ)


		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/phonelock'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3] +'lock'
			dropQ=drop.format(uid)
			cur.execute(dropQ)
		print('Tables destroyed successfully!')

		con.commit()
		con.close()






if __name__ == '__main__':
	main(sys.argv[1:])


