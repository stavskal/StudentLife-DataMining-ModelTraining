import psycopg2,csv,os,sys,getopt


#Standard SQL queries to be used in inserting data
create = """CREATE TABLE appusage (id SERIAL PRIMARY KEY, device VARCHAR(100), time_stamp INT, running_task_base VARCHAR(100), running_task_id INT);"""

insert= """INSERT INTO appusage (device,time_stamp,running_task_base,running_task_id) VALUES (%s,%s,%s,%s);  """

drop = """DROP TABLE appusage"""

query= """ SELECT * FROM appusage WHERE id= %s; """




#function for opening csv files and inserting in DB
def dbInsertData(csvfile,c,cur):
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				data=(str(record['device']),str(record['timestamp']),str(record['RUNNING_TASKS_baseActivity_mPackage']),str(record['RUNNING_TASKS_id']))
				cur.execute(insert,data)


def main(argv):
	global create
	con = None
	#attempting connection to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()

	#if user requested database creation the arg is '-i'
	if sys.argv[1]=='-insert':
		cur.execute(create)
		print('Table created')

		#setting directory to load app_usage information
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/app_usage'

		#inserting all files to database
		for filename in os.listdir(directory):
			filename= directory +'/'+ filename
			dbInsertData(filename,'app_usage',cur)

		print('Done with app_usage')
		con.commit()
		con.close()


	elif sys.argv[1]=='-drop':
		
		cur.execute(drop)
		print('Table Deleted')
		con.commit()
		con.close()


if __name__ == '__main__':
	main(sys.argv[1:])