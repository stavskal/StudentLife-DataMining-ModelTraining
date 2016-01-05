# StudentLife-DataMining
###Colllection of scripts to mine information from StudentLife Dataset

The collected Data is inserted into a PostgreSQL database schema into many different tables.

* ensembleLOSO.py Trains a stacked ensemble mix of Regression and Classification while Cross-Validating with Leave One (Student) Out

* ensembleUserSpecific Trains the same stacked architecture, only this time the model is personalized

* processingFunctions.py provides many usefull functions to construct features

* sleepNNreg.py trains a model to predict length of sleep time
* 
 all the rest scripts are used to insert/drop tables in DB
 
 # requirements
 run `pip install -r requirements.txt` inside a Virtual Environment to  install all requirements
