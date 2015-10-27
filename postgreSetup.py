import psycopg2,csv,os,sys,getopt
from processingFunctions import checkScreenOn
#---------------------------------------------!!!!!!!!!!!------------------------------------------
#---------------------------------------------!!!!!!!!!!!------------------------------------------


#Standard SQL queries to be used in inserting data
create1 = """CREATE TABLE appusage (uid VARCHAR(3), device VARCHAR(100), time_stamp INT, running_task_base VARCHAR(100), running_task_id INT);"""

create2 = """CREATE TABLE users ( device VARCHAR(100) PRIMARY KEY);"""

insert= """INSERT INTO appusage (uid,device,time_stamp,running_task_base,running_task_id) VALUES (%s,%s,%s,%s,%s);  """

insert1= """INSERT INTO users (device) VALUES (%s);  """

drop = """DROP TABLE appusage;"""

query= """ SELECT * FROM appusage WHERE id= %s; """

query1= """ SELECT * FROM users; """

#---------------------------------------------!!!!!!!!!!!------------------------------------------
#---------------------------------------------!!!!!!!!!!!------------------------------------------


#inserts users (csv format) in db
def dbInsertUsers(csvfile,cur):
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				data=[record['device']]
				cur.execute(insert1,data)
				break;	


#function for opening csv files and inserting in DB (user INFO)
#ony inserts applications that were user initiated, not background
def dbInsertData(csvfile,cur):
	a=csvfile.split('_')
	b=a[3]
	uid= b[0:3] #uid is in format 'uXX' with XX E [0,59]
	with open(csvfile,'rb') as inCsv:
			parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
			for record in parsed:
				if checkScreenOn(cur,uid,record['timestamp']):
					data=(uid,str(record['device']),str(record['timestamp']),str(record['RUNNING_TASKS_baseActivity_mPackage']),str(record['RUNNING_TASKS_id']))
					cur.execute(insert,data)

#action series of following script:
# 1)connect to database
# 2) check whether user requested insert/drop
# if insert: all files in 'app_usage' folder are scanned and parsed
# if drop: tables destroyed
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

	#if user requested database creation the arg is '-insert'
	if sys.argv[1]=='-insert':
		cur.execute(create2)
		cur.execute(create1)

		print('Tables created')

		#setting directory to load app_usage information
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/app_usage'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			filename = directory +'/'+ filename

			dbInsertUsers(filename,cur)
			dbInsertData(filename,cur)

		print('Done with app_usage')


		#commiting and closing connection to DB
		con.commit()
		con.close()

	#if user selected '-drop' tables are deleted
	elif sys.argv[1]=='-drop':
		
		cur.execute("DROP TABLE appusage")
		cur.execute("DROP TABLE users")
		
		print('Tables Deleted')

		con.commit()
		con.close()




if __name__ == '__main__':
	main(sys.argv[1:])