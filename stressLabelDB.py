import json,csv,sys,os,psycopg2

#-----------------------------------------------------------------------------------------------
#This script is intended to create user specific tables (postgreSQL) containing their reported
#stress level alongside with the time they did so.
# table name: uXX , XX E [0,59] eg. 'u02', 'u32'
# each row: (timestamp,stress_level), stressed/not stressed = 1/0 
#-----------------------------------------------------------------------------------------------



#I had to do it that way, sorry (apparently I didn't)
create = """CREATE TABLE stress 
			 ( u00 INT,u01 INT,u02 INT,u03 INT,u04 INT,u05 INT,u07 INT,u08 INT,u09 INT,
			 	u10 INT,u12 INT,u13 INT,u14 INT,u15 INT,u16 INT,u17 INT,u18 INT,u19 INT,
			 	u20 INT,u22 INT,u23 INT,u24 INT,u25 INT,u27 INT,u30 INT,u31 INT,u32 INT,
			 	u33 INT,u34 INT,u35 INT,u36 INT,u39 INT,u41 INT,u42 INT,u43 INT,u44 INT,
			 	u45 INT,u46 INT,u47 INT,u49 INT,u50 INT,u51 INT,u52 INT,u53 INT,u54 INT,
			 	u56 INT,u57 INT,u58 INT,u59);"""

create1= "CREATE TABLE {0} (time_stamp INT, stress_level INT) ; "
insert1="INSERT INTO {0} (time_stamp,stress_level) VALUES ({1},{2})	"

insert= """INSERT INTO stress (u00,u01,u02,u03,u04,u05,u07,u08,u09,u10,u12,u13,u14,u15,u16,u17,u18,
								u19,u20,u22,u23,u24,u25,u27,u30,u31,u32,u33,u34,u35,u36,u39,u41,u42,u43,
								u44,u45,u46,u47,u49,u50,u51,u52,u53,u54,u56,u57,u58,u59) 
								VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
										%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);  """


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
			cur.execute(createQ)

			#inserting user specific data into table for
			for i in range(0,len(fullfile)):
				#defending against dirty data
				if 'level' in fullfile[i]:
					#converting from scale 1-5 to binay 0/1 - not stressed/stressed
					if int(fullfile[i]['level'])<4:
						stress=1
					else:
						stress=0>

					insertQ=insert1.format(uid, fullfile[i]['resp_time'], stress)
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


