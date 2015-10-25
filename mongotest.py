from pymongo import MongoClient
import psycopg2

client = MongoClient()
db = client.dataset

current_user_time=[]
current_user_app=[]

create = """CREATE TABLE app_usage
(id VARCHAR(100), device VARCHAR(100), time_stamp INT,
running_task_base VARCHAR(100), running_task_id INT);"""


con = psycopg2.connect(database='dataset', user='stev')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print ver


cursor=db.app_usage.find({}, {'RUNNING_TASKS_baseActivity_mPackage':1,'timestamp':1 ,'_id':0 }).sort([('timestamp',-1)])
print('yoyo')
print(db.app_usage.find({}, {'RUNNING_TASKS_baseActivity_mPackage':1,'timestamp':1 ,'_id':0 }).count())
i=0
#create a list with timestamps and apps of one user
for document in cursor:
	print(int(document['timestamp']))
	if i==0:
		TminusDoc=document
		i = i+1

	current_user_time.append(document['timestamp'])
	current_user_app.append(document['RUNNING_TASKS_baseActivity_mPackage'])


	#exit if user changes, timestamp resets
	if (int(TminusDoc['timestamp'])+1 < int(document['timestamp'])):
		print ('user change')
		print(len(current_user_app))
		exit()
		



	TminusDoc=document

