import json,csv,sys,os,psycopg2


create1= """CREATE TABLE {0} (timeStart INT, timeStop INT) ; """
insert1="INSERT INTO {0} (timeStart,timeStop) VALUES ({1},{2})	;"
drop = "DROP TABLE {0};"


def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()


	if sys.argv[1]=='-insert':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/dark/'

		for filename in os.listdir(directory):

			#creating table for each user with name: 'uXX'
			uid = (filename.split('_'))[1][0:3]
			createQ = create1.format(uid)
			cur.execute(createQ)
			print(filename)
			
			with open(os.path.join(directory,filename),'rb') as inCsv:
				parsed = csv.DictReader(inCsv , delimiter = ',' , quotechar='"')
				for record in parsed:
					insertQ = insert1.format(uid, record['start'], record['end'])
					cur.execute(insertQ)
		con.commit()
		con.close()

		#do stuff
		

	elif sys.argv[1]=='-drop':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/dark'
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]
			dropQ=drop.format(uid)
			cur.execute(dropQ)

		print('Tables destroyed successfully!')

		con.commit()
		con.close()






if __name__ == '__main__':
	main(sys.argv[1:])


