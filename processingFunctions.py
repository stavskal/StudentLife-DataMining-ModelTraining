import datetime

#converts unix timestamp to human readable date (e.g '1234567890' -> '2009-02-14  00:31:30')
def UnixTimeConv(timestamp):
	newTime=str(datetime.datetime.fromtimestamp(int(timestamp)))
	year,time=newTime.split(' ')
	#z=x[2].split(':')
	return (year,time)


a=UnixTimeConv(1234567890)
print(a)