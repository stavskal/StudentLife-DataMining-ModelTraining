import matplotlib.pyplot as pyp
import numpy as np
import seaborn as sns
import pandas as pd
import psycopg2,random
import datetime as dt
from geopy.distance import great_circle
from matplotlib import colors
import six
from scipy.interpolate import spline
from unbalanced_dataset import UnderSampler
from sklearn.cluster import KMeans
import datetime
uids1=['u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','u16','u19']
uids2=['u44','u59','u00','u44']
uids = ['u00','u01','u02','u03','u04','u05','u07','u08','u09','u10','u12','u14','u15','u16','u17','u18','u19','u20','u22','u23','u24',
'u25','u27','u30','u31','u32','u33','u34','u35','u36','u39','u41','u42','u43','u44','u45','u46','u47','u49','u50','u51','u52','u53','u54',
'u56','u57','u58','u59']

def unixTimeConv(timestamps):
	""" converts unix timestamp to human readable date 
		(e.g '1234567890' -> '2009 02 14  00 31 30')
	"""
	splitTimes = np.zeros((len(timestamps), 7),dtype='float32')
	i=0
	#print(timestamps)
	for time in timestamps:
		newTime = str(datetime.datetime.fromtimestamp(int(time)))
		yearDate,timeT = newTime.split(' ')
		year,month,day = str(yearDate).split('-')
		hour,minutes,sec = timeT.split(':')
		splitTimes[i,:] = (year,month,day,hour,minutes,sec,time)
		i += 1
	return(splitTimes)

#from processingFunctions import *
def epochCalc(timestamps):
	""" converts timestamp to epoch (day,evening,night) 
		and returns (epoch,time) tuple
	"""
	splitTimes = unixTimeConv(timestamps)

	epochTimes = []
	for i in range(0,len(splitTimes)):
		hour=int(splitTimes[i,3])
		if hour >10 and hour <=18:
			epoch='day'
		elif hour >0 and hour<=10:
			epoch='night'
		else:
			epoch='evening'
		epochTimes.append((epoch,splitTimes[i,6]))
	return epochTimes

def epoch(timestamp):
	""" converts timestamp to epoch (day,evening,night) 
		and returns (epoch,time) tuple
	"""

	newTime = str(datetime.datetime.fromtimestamp(int(timestamp)))
	yearDate,timeT = newTime.split(' ')
	year,month,day = str(yearDate).split('-')
	hour,minutes,sec = timeT.split(':')

	if hour >10 and hour <=18:
		epoch='day'
	elif hour >0 and hour<=10:
		epoch='night'
	else:
		epoch='evening'

	return epoch

def audioEpochFeats(cur,uid,timestamp):
	uidA = uid +'audio'	
	noise = 0
	cur.execute('SELECT time_stamp, audio FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'.format(uidA,timestamp-86400,timestamp))
	records = cur.fetchall()

	tStart = [item[0] for item in records if item[1]!=3]
	timeEpochs = (epochCalc(tStart)) 

	for i in range(0,len(records)):
		if records[i][1]==2:
			noise += 1

	return(noise)

#connecting to database
def actEpochFeats(cur,uid,timestamp):
	uidA = uid +'act'	
	movement = 0
	cur.execute('SELECT time_stamp, activity FROM {0} WHERE time_stamp >= {1} AND time_stamp<= {2}'.format(uidA,timestamp-86400,timestamp))
	records = cur.fetchall()

	tStart = [item[0] for item in records]
	timeEpochs = (epochCalc(tStart)) 

	for i in range(0,len(records)):
		if records[i][1]==1 or records[i][1]==2:
			movement += 1
	return(movement)

def convEpochFeats(cur,uid,timestamp):
	"""Returns total duration and number of conversations
	   calculated in three epochs (day,evening,night), 6 features total"""

	cur.execute('SELECT * FROM {0} WHERE start_timestamp >= {1} AND end_timestamp<= {2}'.format(uid+'con',timestamp-86400,timestamp))
	records = cur.fetchall()
	totalConvsNight=0
	totalConvTimeN=0

	tStart = [item[0] for item in records]
	tStop = [item[1] for item in records]

	timeEpochs = epochCalc(tStart)
	timeEpochs1 = epochCalc(tStop)

	for i in range(0,len(records)):
		totalConvsNight += 1 
		totalConvTimeN += records[i][1]-records[i][0]
	return(totalConvsNight)



def gpsFeats(cur,uid,timestamp):
	# number of clusters as defined by DBSCAN: 14 + 1 for out of town
	# p will hold the percentage of time spent during previous day in each cluster 
	#variances = np.zeros(2)
	cur.execute("SELECT time_stamp,latitude,longitude FROM {0} WHERE time_stamp>= {1} AND time_stamp<={2}".format(uid+'gpsdata',timestamp-86400,timestamp))
	records = cur.fetchall()
	total_dist_trav=0

	if not records:
		return(np.zeros(1))
	t = [item[0] for item in records]

	timeEpochs = epochCalc(t)
	for i in range(1,len(records)):
		#print(records[i][1],records[i][2])
		# if user is in campus assign him to one of 14 clusters
		# otherwise assign to 15th cluster which stands for 'out-of-town'
		if (records[i][1] > 43.60 and records[i][1] <43.75 and records[i][2] > -72.35 and records[i][2] < -72.2):
			# for every gps coordinate pair calculate the distance from cluster
			# centers and assign to the nearest	
			#print(records[i][1:3],records[i-1][1:3])
			if timeEpochs[i][0] =='night':
				total_dist_trav += great_circle(records[i][1:3],records[i-1][1:3]).meters

	return total_dist_trav


def chargeDur(cur,uid,timestamp):
	totalDur = 0
	uidC = uid+'charge'
	#Getting data from database within day period
	cur.execute('SELECT * FROM {0} WHERE start_timestamp>={1} AND end_timestamp<={2}'.format(uidC, timestamp-86400, timestamp) )
	records = cur.fetchall()

	#timeEpochs holds tuples of timestamps and their according epochs
	tStart = [item[0] for item in records]
#	tStop = [item[1] for item in records]
	timeEpochs = epochCalc(tStart)
	#timeEpochs1 = epochCalc(tStop)

	for i in range(0,len(records)):
		totalDur += records[i][1] - records[i][0]

	return totalDur

def darknessDur(cur,uid,timestamp):
	totalDur = 0
	uidC = uid+'dark'
	#Getting data from database within day period
	cur.execute('SELECT * FROM {0} WHERE timeStart>={1} AND timeStop<={2}'.format(uidC, timestamp-86400, timestamp) )
	records = cur.fetchall()

	#timeEpochs holds tuples of timestamps and their according epochs
	tStart = [item[0] for item in records]
#	tStop = [item[1] for item in records]
	timeEpochs = epochCalc(tStart)
	#timeEpochs1 = epochCalc(tStop)

	for i in range(0,len(records)):
		totalDur += records[i][1] - records[i][0]

	return totalDur

def my_group(x):
	if x>=3 and x<6: 
		x=1
	elif x>=6 and x<9:
		x=2
	else:
		x=3
	return x


try:
	con = psycopg2.connect(database='dataset', user='tabrianos')
	cur = con.cursor()
except psycopg2.DatabaseError as err:
	print('Error %s' % err)
	exit()


rec=[]
noiseList=[]
moveList=[]
convList =[]
darkList=[]
distList=[]
chargeList=[]
for u in uids1:
	user = u+'sleep'
	cur.execute('SELECT time_stamp,hour,rate FROM {0}'.format(user) )
	temp = cur.fetchall()
	for lab in temp:
		noiseList.append(audioEpochFeats(cur,u,lab[0]))
		moveList.append(actEpochFeats(cur,u,lab[0]))
		convList.append(convEpochFeats(cur,u,lab[0]))
		darkList.append(darknessDur(cur,u,lab[0]))
		distList.append(gpsFeats(cur,u,lab[0]))
		chargeList.append(chargeDur(cur,u,lab[0]))
	#cur.execute('SELECT hour,rate FROM {0}'.format(user) )
	rec += temp


#print(len(noiseList),len(rec))
#noise = np.zeros((len(noiseList,1)))

sleep = np.array(rec)
print(sleep)
#m = np.mean(moveList)
#for i in range(0,len(moveList)):
#	if moveList[i]>1000:
	#	moveList[i] = m


last = np.column_stack((rec,chargeList,moveList,convList,darkList,distList))
print('this is my list bitch')
print(last)

#cols =['Timestamp','Hours','Rate']
dfsleep = pd.DataFrame(last)
dfsleep.to_csv('sleepfeats2.csv', sep='\t')
exit()

#last = np.column_stack((rec,chargeList,moveList,convList,darkList,distList))

df = pd.read_csv('sleepfeats.csv', sep='\t', index_col=[0])
df.index = pd.to_datetime(df.index,unit='s')
df = df.sort(['1'])
print(df.head(20))
# good sleep : 1  
# bad sleep : 2
df['2'] = df['2'].map({1: 'good', 2:'good', 3:'bad', 4:'bad'})
df['1'] = df['1'].apply(lambda x: my_group(x))
#df = df.groupby(pd.cut(df['1'], np.arange(3,12.1,3)))
print(df[80:90])
#print(df.tail(30))
exit()
#df = df.loc[df['4']<1000]
ax = sns.boxplot(y='5',x='epoch', hue='2',data=df)
pyp.title('Comparing hours slept,rate and conversation')
pyp.xlabel('Hours slept')
x=[0,1,2]
#time1 = ['3-6','6-9', '9-12']
#pyp.xticks(x,time1)
pyp.ylabel('distance')
#pyp.savefig('joint.png')
fig = ax.get_figure()
fig.savefig('hours_rate_conversation2.png')





"""
X = np.load('plotdata/epochFeats.npy')
y = np.load('plotdata/epochLabels.npy')

first_four = X[1:X.shape[0]/2.5,:]
rest = X[(X.shape[0]/2.5):-1,:]

mean_feat_four = np.mean(first_four,axis=0)

mean_feat_rest = np.mean(rest,axis=0)


for u in uids1:
	cur.execute("SELECT stress_level  FROM {0} ".format(u))
	records = cur.fetchall()

	user = u+'sleep'
	cur.execute('SELECT hour FROM {0}'.format(user) )
	records1 = cur.fetchall()

	print(len(records), len(records1))

#print(mean_feat_rest.shape, mean_feat_four.shape)


allstress = np.zeros((len(uids1),100))

i=0
colors = list(six.iteritems(colors.cnames))
for u in uids1:
	cur.execute("SELECT stress_level  FROM {0} ".format(u))
	records = cur.fetchall()
	#print(records)
	x = [ind for ind,ele in enumerate(records)]
	xnew = np.linspace(x[0], x[-1], 100)
	records = spline(x,records,xnew)
	allstress[i,:] = np.asarray(records).reshape(100)
	print(len(xnew), len(records))
	#pyp.subplot(4,1,i)
	#pyp.plot(xnew,records)
	i+=1
#pyp.xlim(0,100)
#pyp.title('Self perceived stress reports of u59') 
#pyp.savefig('stress_time_all.png')

mean_stress = np.mean(allstress,axis=0)
print(mean_stress.shape)
mean_smooth = np.convolve(mean_stress,np.ones(5)/5,'valid')
#pyp.plot(mean_smooth)
#pyp.title('Mean value of stress reports among 16 students') 
#pyp.savefig('meansmooth.png')





rec=[]
noiseList=[]
moveList=[]
convList =[]
darkList=[]
distList=[]
chargeList=[]
for u in uids1:
	user = u+'sleep'
	cur.execute('SELECT hour,rate,time_stamp FROM {0}'.format(user) )
	temp = cur.fetchall()
	for lab in temp:
		#noiseList.append(audioEpochFeats(cur,u,lab[2]))
		#moveList.append(actEpochFeats(cur,u,lab[2]))
		#convList.append(convEpochFeats(cur,u,lab[2]))
		#darkList.append(darknessDur(cur,u,lab[2]))
		#distList.append(gpsFeats(cur,u,lab[2]))
		chargeList.append(chargeDur(cur,u,lab[2]))
	cur.execute('SELECT hour,rate FROM {0}'.format(user) )
	rec += cur.fetchall()


#print(len(noiseList),len(rec))
#noise = np.zeros((len(noiseList,1)))

sleep = np.array(rec)
print(sleep)
#m = np.mean(moveList)
#for i in range(0,len(moveList)):
#	if moveList[i]>1000:
	#	moveList[i] = m


last = np.column_stack((rec,chargeList))
print('this is my list bitch')
print(last)


dfsleep = pd.DataFrame(last)
dfsleep.to_csv('sleepdist.csv', sep='\t')

print(dfsleep.head(10))
ax = sns.violinplot(x=1,y=0,data=dfsleep)
x=[0,1,2,3]
time1 = ['Very good','Fairly good', 'Fairly bad', 'Very bad']
pyp.xticks(x, time1)
pyp.ylabel('Hours slept')
pyp.xlabel('Rate of sleep')
#ax.savefig('sleep_rate_chargeperiod.png')
fig = ax.get_figure()
fig.savefig('sleep_rate_chargeperiod.png')


ax = sns.boxplot(x=1,y=0,data=dfsleep)
x=[0,1,2,3]
time1 = ['Very good','Fairly good', 'Fairly bad', 'Very bad']
pyp.xticks(x, time1)
pyp.ylabel('Hours slept')
pyp.xlabel('Rate of sleep')
fig = ax.get_figure()
fig.savefig('sleep_rate_hour.png')


sns.set_style('darkgrid')
fi = np.load('featimpor.npy')
x = [i for i in range(0,len(fi))]
ax = sns.barplot(x=x,y=fi)
fig = ax.get_figure()
fig.savefig('featimporta1.png')

tse = [87.1432588052,85.3914820793,83.3015085992,81.8932947942]
x=[0,1,2,3]
time1 = ['33','25','20','10']
pyp.xticks(x, time1)
pyp.title('Agreement of separate classifiers')
pyp.ylabel('Percentage of agreement')
pyp.xlabel('Portion of labeled data used')
ax = sns.tsplot(data=tse)
fig = ax.get_figure()	
fig.savefig('tsagree1.png')

#sns.set(color_codes=True)
fi=[0.04262531,  0.03901671,  0.04068426,  0.01825495,  0.01789869,  0.02558647,0.01785529 , 0.01886639,  0.02596094 , 0.02014726 , 0.01935518 , 0.01862711, 0.01893412 , 0.0184053,   0.03011386,  0.03261616,  0.03354703 , 0.03181524,0.03021566 , 0.03009366 , 0.03474774 , 0.03260319 , 0.03663044 , 0.03213318,0.03419763 , 0.02962987 , 0.03523792 , 0.03469404 , 0.03965231 , 0.04484659,0.0456531 ,  0.03584487 , 0.03350954]



x=['Feeling Great','Feeling Good','A little stressed','Definitely Stressed','Stressed Out']
y=[173,439,492,349,297]

ax = sns.barplot(x=x,y=y)
pyp.ylim(0,600)
pyp.title('ADASYN Oversampling')
fig = ax.get_figure()
fig.savefig('LabelDistoverAfter.png')
exit()

y=np.array([4,30,32.9,21.3,10])
x=['Activity','Screen','Conversation','Co-Location','Audio']
ax = sns.kdeplot(y)
#pyp.ylim(0,592)
pyp.title('Feature importance for each Category of feats.')
pyp.ylabel('No. of Students')
fig = ax.get_figure()
fig.savefig('featimp1.png')
exit()

y=[19.4,19.1,20.4,20.3,20.5]
x=['Activity','Screen','Conversation','Co-Location','Audio']
ax = sns.barplot(x=x,y=y)
#pyp.ylim(0,592)
pyp.title('Feature importance for each Category of feats.')
pyp.ylabel('No. of Students')
fig = ax.get_figure()
fig.savefig('featimp1.png')
exit()



y=[2,2,7,4,1]
x=['50-60','60-70','70-80','80-90','>90']
ax = sns.barplot(x=x,y=y)
#pyp.ylim(0,592)
pyp.title('Leave One Student Out accuracy distribution')
pyp.xlabel('Tolerance Accuracy (%)')
pyp.ylabel('No. of Students')
fig = ax.get_figure()
fig.savefig('LOSOaccDist.png')
exit()



x=['Feeling Great','Feeling Good','A little stressed','Definitely Stressed','Stressed Out']
y=[62,286,292,212,156]

ax = sns.barplot(x=x,y=y)
pyp.ylim(0,592)
pyp.title('Manually balanced distribution of stress responses')
fig = ax.get_figure()
fig.savefig('LabelDistAfter.png')
exit()

con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
cur.execute('SELECT latitude,longitude FROM u33gpsdata')
records = cur.fetchall()



print(len(records))
longitude = [item[1] for item in records ]
latitude = [item[0] for item in records ]

pyp.scatter(latitude,longitude)
pyp.savefig('gpsplot.png')



#heat = np.array([[62.06,63.34, 60.70,60.8 ,54],[60.44,61.14, 55.38, 63.99, 60.05],[67.1,60.6, 60.54,57.52,58.14]])
#print(heat.shape)
uids=['u16','u19','u44','u24','u08','u51','u59','u57','u00','u02','u52','u10','u32','u33','u43','u49','Mean']

#user specific    
x = np.array([67.33,75.79,65.6,79.3,83.5,93.27,43.8,74.64,51.4,66,58.57,56.68,55.97,70.5,60.8,54.4])
x = np.append(x,x.mean())


#LOSO
y = np.array([58.87,78.02,47.12,61.53,78.88,86.36,57.62,68.65,45.94,76.6,58.53,59.43,50,34,55.84,34.4])
y = np.append(y,y.mean())
print(x.shape,y.shape,x.mean(),y.mean())
listaa=[x.min(),y.min(),x.max(),x.max()]

mini = min(listaa)
maxi = max(listaa)
print(listaa)
pyp.scatter(x,y, marker='o')
pyp.xlim(mini-4,maxi+6.7)
pyp.ylim(mini-4,maxi+6.7)
pyp.gca().set_aspect('equal', adjustable='box')
pyp.xlabel('User-Specific Model Accuracy')
pyp.ylabel('Group Model Accuracy (LOSO)')
pyp.title('Comparison of Personalized and Generic Models')

for label, xi, yi in zip(uids,x,y):
	if label!='Mean':
		fc='yellow'
	else:
		fc='red'
	pyp.annotate(
		label,
		xy=(xi,yi), xytext=(0,15),
		textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.3', fc = fc, alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

pyp.savefig('losoUser1.png')

ax1 = sns.heatmap(heat, vmin=50, vmax=65)

pyp.title('Cross-Validation Accuracy Heatmap')

xA = np.array([i for i in range(0,5)])
yA = np.array([i for i in range(0,3)])
ytic = ['100','60','25']
xtic = ['Two days','One day', '3/4 day','Half day', 'Quarter of day']
pyp.xticks(xA, xtic)
pyp.yticks(yA, ytic)
sb = ax1.get_figure()
sb.savefig('heatmap2.png')
print('done baby')

pyp.figure()
y1=np.array([]   [62.98, 58.8, 61.25, 60.07, 60.09, 61.11 ,55.2])
x1 = [0,1,2,3,4,5,6]
xtic = ['100','70','60','50','40','30','20']
pyp.xticks(x1,xtic)
pyp.title('Accuracy performance with User-Specific Time Window')
pyp.xlabel('No. of Most Common Apps kept')
pyp.ylabel('Averaged accuracy over Users (%)')

ax = sns.tsplot(data=y1, estimator=np.median)
fig = ax.get_figure()
fig.savefig('meanStressWinTRY.png')


#               max             		average            			min
y1 = np.array([[70.3,74,69.2,69.2,84.6],[61.4,52.6,54.6,50.1,58.2],[50,31.8,41.66,50,25]])

y = np.array([[90.3,74,69.2,69.2,84.6],[61.4,52.6,54.6,50.1,58.2],[50,31.8,41.66,50,25]])

x=[0,1,2,3,4]
xtic=['120','100','70','50','35']
pyp.xticks(x, xtic)
pyp.title('Accuracy performance with User-Specific Time Window')
pyp.xlabel('No. of Most Common Apps kept')
pyp.ylabel('Accuracy over Users (%)')
ax = sns.tsplot(data=y)
fig = ax.get_figure()
fig.savefig('coolplot.png')




uids1=['u00','u24','u36','u19']
uids2=['u00','u24','u36','u19','u52','u16','u59']


con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()
labels =[]
for u in uids2:
	records = loadStressLabels(cur,u)
	labels += [i[1] for i in records]

print(labels)

x=[0,1]
xtic=['zero', 'one']
pyp.hist(labels)
pyp.savefig('tttt123.png')
ax = sns.distplot(labels, kde=False)
fig = ax.get_figure()	
fig.savefig('ttttt.png')

"""