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

con = psycopg2.connect(database='dataset', user='tabrianos')
cur = con.cursor()

appStats=computeAppStats(cur,'u22',day)
print(len(appStats))


a=random.choice(appStats[1].keys())
b=random.choice(appStats[2].keys())
c=random.choice(appStats[2].keys())
d=random.choice(appStats[3].keys())
e=random.choice(appStats[1].keys())

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

x1=np.linspace(0,10,len(y1))
x2=np.linspace(0,10,len(y2))
x3=np.linspace(0,10,len(y3))
x4=np.linspace(0,10,len(y4))
x5=np.linspace(0,10,len(y5))
pyp.plot(x1,y1,'*--')
pyp.plot(x2,y2,'^--')
pyp.plot(x3,y3,'+--')
pyp.plot(x4,y4,'o--')
pyp.plot(x5,y5,'x--')
pyp.ylabel("""Application Usage (%total)""")
pyp.xlabel("""Week""")
pyp.savefig('myplot.png')
