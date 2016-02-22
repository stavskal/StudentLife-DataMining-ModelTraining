import json,csv,sys,os,psycopg2,random
import numpy as np
from collections import Counter 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.cross_validation import cross_val_predict, StratifiedKFold, KFold,cross_val_score, LeaveOneLabelOut
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import preprocessing
from unbalanced_dataset import UnderSampler, ClusterCentroids
import matplotlib.pyplot as plt
import time
import warnings





def main(argv):
	#Change to parent directory to load data
	#os.chdir(os.path.pardir)
	X=np.load('data/sleephourlyX.npy')
	Y=np.load('data/sleephourly_hours.npy')
	X = np.delete(X,0,0)
	Y = np.delete(Y,0,0)
	print(Counter(Y))
#	ind = np.random.permutation(X.shape[0])
#	Y = Y[ind]
#	X = X[ind,:]
	feats = [i for i in range(0,70)]
	X = X[:,feats]
	print(X.shape)

	for i in range(0,Y.shape[0]):
		if Y[i]>=3 and Y[i]<=5.5:
			Y[i]=0
		elif Y[i]>5.5 and Y[i]<9:
			Y[i]=1
		else:
			Y[i]=2
	
	#cc = UnderSampler()
	#X,Y = cc.fit_transform(X,Y)
	print(Counter(Y))

	if sys.argv[1]=='-ensemble':
		RF  = []
		outputRF = []
		outRFtest=[]
		totalRF=0
		totalRFR=0



		print(X.shape,Y.shape)
		n_folds=3
		skf = StratifiedKFold(Y,n_folds=n_folds)
		kf = KFold(X.shape[0],n_folds=n_folds,shuffle=True)
		for traini, testi in skf:
			rfr= RandomForestClassifier()
			rf1= RandomForestClassifier(n_estimators=300,class_weight='auto',n_jobs=-1,criterion='entropy',min_samples_split=1)

			rfr.fit(X[traini],Y[traini])
			rf1.fit(X[traini],Y[traini])
			print(Counter(Y[traini]))
			pred = rfr.predict(X[testi])
			predR = rf1.predict(X[testi])
			print(Counter(predR),Counter(pred),Counter(Y[testi]))


			print(recall_score(Y[testi],pred, average=None))
			print(recall_score(Y[testi],predR, average=None))
			
			totalRF += accuracy_score(Y[testi],pred)
			totalRFR += accuracy_score(Y[testi],predR)


		print('True RF: {0}'.format(totalRF/n_folds))
		print('True RFR: {0}'.format(totalRFR/n_folds))
	#	print('True XGB: {0}'.format(totalXGB/n_folds))
		#print('True Ada: {0}'.format(totalada/n_folds))






if __name__ == '__main__':
	main(sys.argv[1:])