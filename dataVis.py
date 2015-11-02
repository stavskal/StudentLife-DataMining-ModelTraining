import matplotlib.pyplot as pyp
import numpy as np
import seaborn as sns
import psycopg2,random
import datetime as dt
from processingFunctions import *


heat = np.array([[63.34, 60.70,60.8 ,54],[61.14, 55.38, 63.99, 60.05],[60.6, 60.54,57.52,58.14]])
print(heat)
ax1 = sns.heatmap(heat, vmin=50, vmax=65)
sb = ax1.get_figure()
sb.savefig('heatmap.png')
print('done baby')


#x = [0,2,4,6,8]
#y = [0,3,3,7,1]
#pyp.plot(x,y)
#pyp.savefig('myplot.png')

y1=[]
y2=[]
y3=[]
y4=[]
y5=[]
y6=[]
y7=[]



xA = np.array([i for i in range(0,4)])
yA = np.array([67.37, 59.10 , 58.79 , 59.28])
xtic = ['One day', '3/4 day','Half day', 'Quarter of day']
pyp.xticks(xA, xtic)
ax = sns.tsplot(yA,xA)
sns.set(style = 'ticks', color_codes=True)
sns.axes_style('darkgrid')
pyp.title('Random Forest Personalized Classifier(100 most common apps)')
pyp.ylabel('Averaged Accuracy over Users (%)')
pyp.xlabel('Time period for App Statistics calculation')
seabornFig = ax.get_figure()
seabornFig.savefig('testSea.png')
#pyp.plot(xA,yA,'--x')

#pyp.savefig('trainingTimes.png')



"""
stressL = []


stressL = []
epochs = []


con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()

#appStats = computeAppStats(cur,'u01',day)
stressLabels1 = loadStressLabels(cur,'u01')
stressLabels1 = sorted(stressLabels1, key=lambda x:x[0] )

stressLabels2 = loadStressLabels(cur,'u02')
stressLabels2 = sorted(stressLabels2, key=lambda x:x[0] )

stressLabels3 = loadStressLabels(cur,'u09')
stressLabels3 = sorted(stressLabels3, key=lambda x:x[0] )

uids=['u12','u16','u17','u30','u31']
for i in uids:

	temp = loadStressLabels(cur,i)
	temp = sorted(temp, key=lambda x:x[0])

	stressL.append(temp)

for i in uids:
	
	temp = epochStressNorm(cur,i)
	epochs.append(temp)

	y1.append(np.linspace(0,10,len(temp)))


for i in range(0,len(uids)):
	if len(epochs[i])>5:
		pyp.plot(y1[i],epochs[i])
pyp.savefig('stressEpochs.png')


y=[]
newL=[]

for i in range(0,len(uids)):

	temp1= np.array( [int(j[0]) for j in stressL[i]] )
	dates=[dt.datetime.fromtimestamp(ts) for ts in temp1]
	newL.append(dates)

	stressLab=np.array([int(j[1]) for j in stressL[i]])

	y.append(stressLab)


#xStress = np.sort(xStress)
#xStress = sorted(singleAppOccur, key=lambda x:x[1] )

lStress1= np.array([int(i[1]) for i in stressLabels1])
lStress2= np.array([int(i[1]) for i in stressLabels2])
lStress3= np.array([int(i[1]) for i in stressLabels3])


for i in range(0,len(uids)):
	pyp.plot(newL[i],y[i])

pyp.savefig('stressLabels.png')



"""