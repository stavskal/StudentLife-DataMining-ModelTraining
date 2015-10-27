import matplotlib.pyplot as pyp
import numpy as np
import psycopg2,random
from processingFunctions import *

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


con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()

appStats = computeAppStats(cur,'u01',day)
stressLabels = loadStressLabels(cur,'u22')
stressLabels = sorted(stressLabels, key=lambda x:x[0] )


a=random.choice(appStats[1].keys())
b=random.choice(appStats[2].keys())
c=random.choice(appStats[2].keys())
d=random.choice(appStats[3].keys())
e=random.choice(appStats[4].keys())
f=random.choice(appStats[4].keys())
g=random.choice(appStats[1].keys())

print(len(appStats))



for i in range(0,len(appStats)):

	if a in appStats[i].keys():
		y1.append(appStats[i][a])
	if b in appStats[i].keys():
		y2.append(appStats[i][b])
	if c in appStats[i].keys():
		y3.append(appStats[i][c])
	if d in appStats[i].keys():
		y4.append(appStats[i][d])
	if e in appStats[i].keys():
		y5.append(appStats[i][e])
	if f in appStats[i].keys():
		y6.append(appStats[i][f])
	if g in appStats[i].keys():
		y7.append(appStats[i][g])


x1=np.linspace(0,10,len(y1))
x2=np.linspace(0,10,len(y2))
x3=np.linspace(0,10,len(y3))
x4=np.linspace(0,10,len(y4))
x5=np.linspace(0,10,len(y5))
x6=np.linspace(0,10,len(y6))
x7=np.linspace(0,10,len(y7))
pyp.figure()
pyp.plot(x1,y1,'*--')
pyp.plot(x2,y2,'^--')
pyp.plot(x3,y3,'+--')
pyp.plot(x4,y4,'o--')
pyp.plot(x5,y5,'x--')
pyp.plot(x6,y6,'x--')
pyp.plot(x7,y7,'x--')
pyp.ylabel("""Application Usage (%total)""")
pyp.xlabel("""Week""")
pyp.savefig('myplot3.png')

pyp.figure()
xStress = np.array([int(i[0]) for i in stressLabels])
xStress = xStress - np.amin(xStress)
xStress = xStress / 10000
#xStress = np.sort(xStress)
#xStress = sorted(singleAppOccur, key=lambda x:x[1] )
print(xStress)
lStress= np.array([int(i[1]) for i in stressLabels])


pyp.plot(xStress,lStress,'*--')
pyp.savefig('stress.png')
