import json,csv,sys,os,psycopg2

#-----------------------------------------------------------------------------------------------
#This script is intended to create user specific tables (postgreSQL) containing their reported
#stress level alongside with the time they did so.
# table name: uXX , XX E [0,59] eg. 'u02', 'u32'
# each row: (timestamp,stress_level), stressed/not stressed = 1/0 
#-----------------------------------------------------------------------------------------------


create1= "CREATE TABLE {0} (time_stamp INT, stress_level INT) ; "
insert1="INSERT INTO {0} (time_stamp,stress_level) VALUES ({1},{2})	"



drop = "DROP TABLE {0};"
#end of paranoia



def main(argv):
	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()



	#depending on what user chose, either inserts, or drops tables
	if sys.argv[1]=='-insert':
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
					if int(fullfile[i]['level'])==4 or int(fullfile[i]['level'])==5:
						stress = 0
					else:
						stress = 1
					

					insertQ=insert1.format(uid, fullfile[i]['resp_time'], stress)
					print(insertQ)
					cur.execute(insertQ)
		con.commit()
		con.close()


	#destroying all user tables
	elif sys.argv[1]=='-drop':

		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/EMA/response/Stress'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		print('Tables destroyed successfully!')

		con.commit()
		con.close()

















if __name__ == '__main__':
	main(sys.argv[1:])


