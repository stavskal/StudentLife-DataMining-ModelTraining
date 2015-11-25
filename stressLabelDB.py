import json,csv,sys,os,psycopg2

#-----------------------------------------------------------------------------------------------
#This script is intended to create user specific tables (postgreSQL) containing their reported
#stress level,mood,sleep alongside with the time they did so.
# table name: uXX , XX E [0,59] eg. 'u02', 'u32'
# each row: (timestamp,stress_level), stressed/not stressed = 1/0 
#-----------------------------------------------------------------------------------------------


create1= "CREATE TABLE {0} (time_stamp INT, stress_level INT) ; "
insert1="INSERT INTO {0} (time_stamp,stress_level) VALUES ({1},{2})	"

create2= "CREATE TABLE {0} (time_stamp INT, mood INT) ; "
insert2="INSERT INTO {0} (time_stamp,mood) VALUES ({1},{2})	"

create3= "CREATE TABLE {0} (hour FLOAT, rate INT, time_stamp INT) ; "
insert3="INSERT INTO {0} (hour,rate,time_stamp) VALUES ({1},{2},{3});"




drop = "DROP TABLE {0};"



def main(argv):
	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()



	#depending on what user chose, either inserts, or drops tables
	if sys.argv[1]=='-stress':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/Stress'

		#iterating through all json files
		for filename in os.listdir(directory):
			try:
				jsonData=open(os.path.join(directory,filename))
			except(IOError, RuntimeError):
				print("Could not open file : %s" % filename)

			fullfile=json.load(jsonData)
			
			#creating table for each users' reported stress level along with timestamps
			uid=(filename.split('_'))[1][0:3]

			createQ=create1.format(uid)
			print(createQ)
			cur.execute(createQ)

			#inserting user specific data into table for
			for i in range(0,len(fullfile)):
				#defending against dirty data
				if 'level' in fullfile[i]:
					#converting from scale 1-5 to binay 0/1 - not stressed/stressed
					if int(fullfile[i]['level'])==5:
						stress = 0
					elif int(fullfile[i]['level'])==4:
						stress = 1
					elif int(fullfile[i]['level'])==1:
						stress = 2
					elif int(fullfile[i]['level'])==2:
						stress = 3
					elif int(fullfile[i]['level'])==3:
						stress = 4

					insertQ=insert1.format(uid, fullfile[i]['resp_time'], fullfile[i]['level'])
					print(insertQ)
					cur.execute(insertQ)
		


	
		

	elif sys.argv[1]=='-mood':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/PAM'

		#iterating through all json files
		for filename in os.listdir(directory):
			try:
				jsonData=open(os.path.join(directory,filename))
			except(IOError, RuntimeError):
				print("Could not open file : %s" % filename)

			fullfile=json.load(jsonData)
			
			#creating table for each users' reported stress level along with timestamps
			uid=(filename.split('_'))[1][0:3]+'mood'

			createQ=create2.format(uid)
			print(createQ)
			cur.execute(createQ)

			#inserting user specific data into table for
			for i in range(0,len(fullfile)):
				#defending against dirty data
				if 'picture_idx' in fullfile[i]:
					#converting from scale 1-5 to binay 0/1 - not stressed/stressed
					if int(fullfile[i]['picture_idx'])>=1 and int(fullfile[i]['picture_idx'])<=4:
						mood = 0
					elif int(fullfile[i]['picture_idx'])>=5 and int(fullfile[i]['picture_idx'])<=8:
						mood = 1
					elif int(fullfile[i]['picture_idx'])>=9 and int(fullfile[i]['picture_idx'])<=12:
						mood = 2
					else:
						mood = 3
					

					insertQ=insert2.format(uid, fullfile[i]['resp_time'], mood)
					print(insertQ)
					cur.execute(insertQ)



	elif sys.argv[1] == '-sleep':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/Sleep'

		#iterating through all json files
		for filename in os.listdir(directory):
			try:
				jsonData=open(os.path.join(directory,filename))
			except(IOError, RuntimeError):
				print("Could not open file : %s" % filename)

			fullfile=json.load(jsonData)
			
			#creating table for each users' reported stress level along with timestamps
			uid=(filename.split('_'))[1][0:3]+'sleep'

			createQ=create3.format(uid)
			print(createQ)
			cur.execute(createQ)

			for i in range(0,len(fullfile)):
				if 'hour' and 'resp_time' and 'rate' in fullfile[i]:
					timeslept = 2.5 +0.5*float(fullfile[i]['hour'])
					if timeslept==2.5:
						print(fullfile[i]['hour'])
					else:
						insertQ=insert3.format(uid, timeslept, fullfile[i]['rate'],fullfile[i]['resp_time'])
					#print(insertQ)
						cur.execute(insertQ)





	#destroying all user tables
	elif sys.argv[1]=='-drop':

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/Stress'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/Sleep'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]+'sleep'
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		print('Tables destroyed successfully!')







	con.commit()
	con.close()
















if __name__ == '__main__':
	main(sys.argv[1:])


