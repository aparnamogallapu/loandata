# -*- coding: utf-8 -*-
"""
Created on Thu May  9 06:33:07 2019

@author: HP
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.getcwd()
os.chdir("F:\\ds")
loan_data=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\train.csv")
loan_data.shape
data=loan_data
loan_data.describe()
loan_data.info()
loan_data.isna().sum()
loan_data.duplicated()
loan_data.isna().sum().sort_values(ascending=False)

%matplotlib inline
loan_data.isna().sum().plot.bar()

loan_data.columns
Loan_ID=loan_data.Loan_ID
loan_data.drop(["Loan_ID"],axis=1,inplace=True)
loan_data["Gender"].value_counts()
loan_data["Gender"].unique()
loan_data.Gender.isna().sum()
loan_data.Gender.mode()
loan_data["Gender"]=loan_data.Gender.astype("str").transform(lambda x: x.replace('nan','Male'))
#loan_train["Gender"]=loan_train.Gender.astype('str').transform(lambda x: x.replace('nan','Male'))
loan_data.Gender.isna().sum()
loan_data["Married"].value_counts()
loan_data["Married"].unique()
loan_data.Married.isna().sum()
loan_data["Married"]=loan_data.Married.astype("str").transform(lambda x: x.replace("nan","Yes"))
loan_data.Married.isna().sum()
loan_data["Dependents"].value_counts()
loan_data["Dependents"].unique()
loan_data.Dependents.isna().sum()
loan_data["Dependents"]=loan_data["Dependents"].replace('3+',"3")
loan_data["Dependents"]=loan_data.Dependents.astype("str").transform(lambda x: x.replace("nan",'0'))
loan_data.Dependents.isna().sum()
loan_data["Self_Employed"].value_counts()
loan_data["Self_Employed"].unique()
loan_data.Self_Employed.isna().sum()
#loan_data.["Self_Employed"]=loan_data.Self_Employed.astype("str").transform(lambda x: x.replace('nan','No'))
loan_data["Self_Employed"]=loan_data.Self_Employed.astype('str').transform(lambda x: x.replace('nan','No'))
loan_data.Self_Employed.isna().sum()
loan_data.LoanAmount.mean()
loan_data["LoanAmount"]=loan_data["LoanAmount"].fillna(146)
loan_data.LoanAmount.isna().sum()
loan_data.Loan_Amount_Term.mode()

loan_data["Loan_Amount_Term"]=loan_data.Loan_Amount_Term.astype("str").transform(lambda x:x.replace('nan','360.0'))
loan_data.Loan_Amount_Term.isna().sum()
loan_data["Credit_History"].unique()
loan_data["Credit_History"].value_counts()
loan_data["Credit_History"]=loan_data.Credit_History.astype("str").transform(lambda x:x.replace("nan","1.0"))
loan_data.info()
loan_data.isna().sum()
loan_data.Credit_History.value_counts()
loan_data["Credit_History"]=loan_data.Credit_History.astype("str").transform(lambda x:x.replace("nan","1.0"))



########EDA####################

%matplotlib
loan_data.Self_Employed.value_counts().sort_values(ascending=False).plot.barh()

loan_data.Gender.value_counts().sort_values(ascending=False).plot.bar()
loan_data.Married.value_counts().sort_values(ascending=False).plot.bar()
loan_data.Dependents.value_counts().sort_values(ascending=False).plot.bar()

loan_data.Education.value_counts().sort_values(ascending=False).plot.barh()
loan_data.Self_Employed.value_counts().sort_values(ascending=False).plot.barh()
loan_data.Credit_History.value_counts().sort_values(ascending=False).plot.barh()

loan_data.Property_Area .value_counts().sort_values(ascending=False).plot.barh()


loan_data.groupby(['Education','Credit_History'])['Education'].count().plot.bar()
#loan_data.groupby(['Dependents','Credit_History'])['Dependents'].count().plot.bar()
#loan_data.groupby(['Property_Area','Credit_History'])['Property_Area'].count().plot.bar()
loan_data.groupby(['Loan_Status','Gender'])['Loan_Status'].count().plot.bar()
loan_data.groupby(['Loan_Status','Married'])['Loan_Status'].count().plot.bar()
loan_data.groupby(['Loan_Status','Education'])['Loan_Status'].count().plot.bar()
loan_data.groupby(['Loan_Status','Dependents'])['Loan_Status'].count().plot.bar()




################cont..preprocessing################
loan_data["Gender"]=loan_data.Gender.map({"Male": 1, "Female": 0})
loan_data["Married"]=loan_data.Married.map({"Yes": 1, "No": 0})
loan_data[loan_data.isnull()]
#loan_data["Married"]=loan_data["Married"].fillna(1)
loan_data["Education"]=loan_data.Education.map({"Graduate":1,"Not Graduate":0})
loan_data["Self_Employed"]=loan_data.Self_Employed.map({"Yes": 1, "No": 0})
loan_data["Loan_Amount_Term"]=loan_data.Loan_Amount_Term.astype("float")
loan_data["Loan_Amount_Term"]=loan_data.Loan_Amount_Term.astype("int")


loan_data["Credit_History"]=loan_data.Credit_History.astype("float")
loan_data["Credit_History"]=loan_data.Credit_History.astype("int")
loan_data["Dependents"]=loan_data["Dependents"].astype("float")
loan_data["Dependents"]=loan_data["Dependents"].astype("int")
loan_data["Loan_Status"]=loan_data.Loan_Status.map({"Y": 0, "N": 1})
loan_data.info()
loan_data.columns



##############    CREATING DUMMY VARIABLES #######################

loan_data_dum=pd.get_dummies(loan_data)

catcols=loan_data.select_dtypes(["object"])
for cat in catcols:
    print(cat)
    print(loan_data[cat].value_counts())
    print("--"*20)

loan_data_dum.columns


############ SPLITTING THE DATA AS DEPENDENT(Y) AND INDEPENDENT(X) #################

y=loan_data_dum.Loan_Status.values

loan_data_dum.drop(['Loan_Status'],axis=1,inplace=True)


x=loan_data_dum.values

############## SCALING THE DATA ##################### OBSERVE DATAFRAME BEFOR N AFTER SCALING ########## 

from sklearn.preprocessing import StandardScaler


scaler = StandardScaler().fit(x)

x=scaler.transform(x)

############# CHECKING WHETHER DATA SCALED OR NOT ############
x[:6,:]
y[:]


################ SPLITTING THE DATA AS TRAIN AND TEST #################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.215,random_state=0)


############ IMPORTING LOGOSTIC REGRESSION ALGORITHM ##################

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

########## CREATED OBJECT AS MODEL AND USING LOGOSTIC REGRESSION CLASS ##########
model=LogisticRegression()

model.fit(x,y)
model.score(x,y)############0.8110749185667753

model.fit(x_test,y_test)

model.score(x_test,y_test)############## 0.8378378378378378

model.score(x_train,y_train)########## 0.7948717948717948
################ APPLYING KFOLD I.E CROSS VALIDATION ##############3

from sklearn.model_selection import cross_val_score,KFold

kfold=KFold(n_splits=12,random_state=0)
score=cross_val_score(model,x,y,cv=kfold,scoring="accuracy")
score.mean()#######0.8062154348919055
print('Score:',score.mean)
print('Score:',score.mean())########## 0.8062154348919055


################prediting the Test Set result#################
y_pred = model.predict(x_test)
y_pred

#######Making the Confusion Matrix#############
from sklearn.metrics import confusion_matrix####array([[132,   2],
                                                       #[ 28,  23]], dtype=int64)
cm=confusion_matrix(y_test,y_pred)
cm

############IMPORT SVM#############
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

########## CREATED OBJECT AS MODEL AND USING SVC CLASS ##########
model=SVC()
model.fit(x,y)
model.score(x,y)############0.8241
model.fit(x_test,y_test)
model.score(x_test,y_test)############## 0.8721
model.score(x_train,y_train)########## 0.8004
################ APPLYING KFOLD I.E CROSS VALIDATION ##############3

from sklearn.model_selection import cross_val_score,KFold

kfold=KFold(n_splits=12,random_state=0)
score=cross_val_score(model,x,y,cv=kfold,scoring="accuracy")
score.mean()#######0.8078
print('Score:',score.mean)
print('Score:',score.mean())########## 0.8078


################prediting the Test Set result#################
y_pred = model.predict(x_test)
y_pred

#######Making the Confusion Matrix#############
from sklearn.metrics import confusion_matrix####array([[95,   1],
                                                       #[ 16,  21]], dtype=int64)
cm=confusion_matrix(y_test,y_pred)
cm



loan_data.drop(["Loan_Status"],axis=1,inplace=True)

loan_data['Loan_ID']=Loan_ID

############################################# WORKING ON TEST DATA ##########################################
loan_test=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\test.csv")
loan_test.shape
loan_test.columns
data2=loan_test.copy()
loan_test.info()
Loan_ID=loan_test.Loan_ID
#Loan_ID_test=loan_test.Loan_ID
loan_test.drop(["Loan_ID"],axis=1,inplace=True)
loan_test.isna().sum()
loan_test.Gender.value_counts()
loan_test["Gender"]=loan_test.Gender.astype("str").transform(lambda x:x.replace('nan','Male'))
loan_test["Gender"]=loan_test.Gender.map({"Male":1,"Female":0})
loan_test.Gender.isna().sum()
loan_test.Dependents.value_counts()
loan_test.Dependents.unique()
#loan_test.Dependents.mode()
loan_test["Dependents"]=loan_test["Dependents"].replace("3+","3")
loan_test["Dependents"]=loan_test.Dependents.astype("str").transform(lambda x:x.replace('nan','0'))
loan_test["Dependents"]=loan_test["Dependents"].astype("float")
loan_test["Dependents"].unique()
loan_test["Dependents"]=loan_test["Dependents"].astype("int")
loan_test.Self_Employed.value_counts()
loan_test["Self_Employed"]=loan_test.Self_Employed.astype("str").transform(lambda x:x.replace("nan","No"))
loan_test["Self_Employed"]=loan_test.Self_Employed.map({"Yes":1,"No":0})
loan_test.LoanAmount.value_counts()
loan_test.LoanAmount.mean()
loan_test["LoanAmount"]=loan_test["LoanAmount"].fillna(136)
#loan_test.Loan_Amount_Term.value_counts()
loan_test["Loan_Amount_Term"]=loan_test.Loan_Amount_Term.astype("str").transform(lambda x:x.replace('nan','360.0'))
loan_test["Loan_Amount_Term"]=loan_test["Loan_Amount_Term"].astype("float")
loan_test["Loan_Amount_Term"]=loan_test["Loan_Amount_Term"].astype("int")
loan_test.Credit_History.value_counts()
loan_test["Credit_History"]=loan_test.Credit_History.astype("str").transform(lambda x:x.replace('nan','1.0'))
loan_test["Credit_History"]=loan_test["Credit_History"].astype("float")
loan_test["Credit_History"]=loan_test["Credit_History"].astype("int")
#loan_test["Credit_History"]=loan_test["Credit_History"].astype("str")

loan_test.info()

loan_test["Education"]=loan_test.Education.map({"Graduate": 1, "Not Graduate": 0})
loan_test["Married"]=loan_test.Married.map({"Yes": 1, "No": 0})
loan_test.Married.isna().sum()



##############    CREATING DUMMY VARIABLES #######################
loan_test_dum=pd.get_dummies(loan_test)

list(loan_test_dum.columns)

X_test=loan_test_dum.values
X_test=scaler.transform(X_test)

'''Property_Area_test_dum=pd.get_dummies(loan_test["Property_Area"])
loan_test=pd.concat([loan_test,Property_Area_test_dum],axis=1)
loan_test.drop(["Property_Area"],axis=1,inplace=True)'''


'''loan_test_dum=pd.get.dummies(loan_test)
catcols=loan_test.select_dtypes(['object'])
for cat in catcols:
    print(cat)
    print(loan_test[cat].value_counts())
    print('--'*20)
loan_test.info()
loan_test_dum.columns'''

list((loan_data_dum.columns==loan_test_dum.columns)) 
(loan_data_dum.columns==loan_test_dum.columns) 
    
pd.value_counts(list((loan_data_dum.columns==loan_test_dum.columns)))

model.predict(X_test)

submission_predict=model.predict(X_test)
#submission_loan=np.exp(model_log.predict(X_test))
submission_loan_predicted=pd.DataFrame(data={'Loan_ID':Loan_ID.values,'Loan_Status':submission_predict})
submission_loan_predicted["Loan_Status"]=submission_loan_predicted["Loan_Status"].map({1: "N", 0: "Y"})


import os

os.chdir('F:\\ds')

submission_loan_predicted.to_csv('Sample_Submission_loan.csv',index = False)

loan_submission=pd.read_csv('Sample_Submission_loan.csv')
loan_submission.columns

loan_test['Loan_ID']=Loan_ID

(loan_data.columns==loan_test.columns)


loan_data.columns

loan_test.columns

loan_test.info()









