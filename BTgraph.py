import json,csv,sys,os,psycopg2
from netaddr import EUI

create = "CREATE TABLE {0} (time_stamp INT, mac VARCHAR(100), class_id INT, level INT) ;"
insert = "INSERT INTO {0} (time_stamp, mac, class_id,level) VALUES ({1},{2},{3},{4}) ;"
drop = "DROP TABLE {0}"

# IMPROTANT NOTE
# MAC addresses are stored in their oct() form as strings



# Parses data from CSV files to insert bluetooth scans in database tables
def dbInsertBTscan(csvfile,cur):
	uid=(csvfile.split('_'))[1][0:3] +'bt'
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
					



def main(argv):

	#connecting to database
	try:
		con = psycopg2.connect(database='dataset', user='tabrianos')
		cur = con.cursor()

	except psycopg2.DatabaseError as err:
		print('Error %s' % err)
		exit()





	if sys.argv[1]=='-insert':
		#setting directory to load app_usage information
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/bluetooth'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			filename = directory +'/'+ filename

			dbInsertBTscan(filename,cur)

		con.commit()
		con.close()
		#do stuff
		

	elif sys.argv[1]=='-drop':
		directory = os.path.dirname(os.path.abspath(__file__)) + '/dataset/sensing/bluetooth'

		#inserting all files to database from given directory
		for filename in os.listdir(directory):
			uid=(filename.split('_'))[1][0:3]+'bt'
			dropQ=drop.format(uid)
			cur.execute(dropQ)c
		con.commit()
		con.close()
		#do stuff





if __name__ == '__main__':
	main(sys.argv[1:])