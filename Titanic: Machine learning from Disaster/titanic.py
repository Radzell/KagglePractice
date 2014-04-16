# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, svm
from sklearn.preprocessing import Imputer
import math
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import accuracy_score

# <codecell>

columns = ['Pclass','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

# <codecell>

columns = ['Pclass','Sex','Age','SibSp','Parch'];

# <codecell>

#columns = ['Fare','Age']

# <codecell>

train_df = pd.read_csv("train.csv")
#vectorize training data
train_df['Sex']=train_df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)

#fills in the na's
#and int columns won't have NaNs else they would be upcast to float.
for col in train_df:
    dt = train_df[col].dtype
    
    #fill in Nana
    if dt== float:
        train_df[col].fillna(train_df[col].mean(), inplace=True)
    #normalize data
    if dt==float or dt==int:    
        train_df[col] -= train_df[col].max()/2
        train_df[col] /=max( train_df[col].max(),math.fabs(train_df[col].max()))

# <codecell>

train_df.ix[:,columns].head()

# <codecell>

test_df = pd.read_csv("test.csv")

#vectorize training data
test_df['Sex']=test_df["Sex"].apply(lambda sex: 0 if sex == "male" else 1)

#fills in the na's
#and int columns won't have NaNs else they would be upcast to float.
for col in test_df:
    dt = test_df[col].dtype
    #fill in nan
    if dt== float:
        test_df[col].fillna(test_df[col].mean(), inplace=True)
    #normalize data
    if (dt==float or dt==int) and col != 'PassengerId':    
        test_df[col]-=test_df[col].max()/2
        test_df[col] /=max( test_df[col].max(),math.fabs(test_df[col].max()))


clf = svm.SVC(kernel='rbf', degree=3,gamma=9)
model = clf.fit(train_df.ix[:,columns],train_df.ix[:,'Survived'])

y_pred=clf.predict(test_df.ix[:,columns])
test_df["Survived"] = pd.Series(y_pred)
test_df.to_csv("result.csv", cols=['PassengerId', 'Survived'], index=False)

print train_df.head()


print test_df.head()



# <codecell>

#create a list of the types of kerneks we will use for your analysis
types_of_kernels = ['linear', 'rbf', 'poly']


minimum={"error":1,"model":""}
error_rate=0
types_of_kernels = ['linear', 'rbf', 'poly']
model=""
#create crossvalidation sets to test for accurancy
cv =ShuffleSplit(len(train_df.index), train_size=300,test_size=None)
for traincv, testcv in cv:
    for fig_num, kernel in enumerate(types_of_kernels):
        print kernel
        if kernel == 'linear':
            for i in range(1,10):
                # fit the model
                clf = svm.SVC(kernel=kernel, degree=i,gamma=3)
                model = clf.fit(train_df.ix[traincv,columns],train_df.ix[traincv,'Survived'])
                
                 #predict the test set
                y_pred = clf.predict(train_df.ix[testcv,columns])
                y_test = train_df.ix[testcv,'Survived']
                accuracy = accuracy_score(y_test, y_pred)
                error_rate =  1-accuracy
                if error_rate < minimum['error']:
                    minimum['error']=error_rate
                    minimum['model']=model
        if kernel == 'poly':
            for i in range(1,5):
                print i
                # fit the model
                clf = svm.SVC(kernel=kernel, degree=i,gamma=3)
                model = clf.fit(train_df.ix[traincv,columns],train_df.ix[traincv,'Survived'])
                
                 #predict the test set
                y_pred = clf.predict(train_df.ix[testcv,columns])
                y_test = train_df.ix[testcv,'Survived']
                accuracy = accuracy_score(y_test, y_pred)
                error_rate =  1-accuracy
                if error_rate < minimum['error']:
                    minimum['error']=error_rate
                    minimum['model']=model
        if kernel == 'rbf':
            for i in range(1,10):
                # fit the model
                clf = svm.SVC(kernel=kernel, degree=3,gamma=i)
                model = clf.fit(train_df.ix[traincv,columns],train_df.ix[traincv,'Survived'])
                
                 #predict the test set
                y_pred = clf.predict(train_df.ix[testcv,columns])
                y_test = train_df.ix[testcv,'Survived']
                accuracy = accuracy_score(y_test, y_pred)
                error_rate =  1-accuracy
                if error_rate < minimum['error']:
                    minimum['error']=error_rate
                    minimum['model']=model
        print minimum
    break
# <codecell>

print minimum

# <codecell>

import sklearn.ensemble
#create crossvalidation sets to test for accurancy
for traincv, testcv in cv:
    # fit the model
    clf = sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0)
    #clf = svm.SVC(kernel=kernel, degree=3,gamma=3)
    clf.fit(train_df.ix[traincv,columns],train_df.ix[traincv,'Survived'])
    
     #predict the test set
    y_preds = clf.predict(train_df.ix[testcv,columns])
    y_test = train_df.ix[testcv,'Survived']
    accuracy = accuracy_score(y_test, y_pred)
    error_rate =  1-accuracy
    if error_rate<minimum['error']:
        minimum['error']=error_rate
        minimum['model']=model
    print '{}'.format(error_rate)
    break
print minimum

# <codecell>
from sklearn import  neighbors, linear_model
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

knn = neighbors.KNeighborsClassifier(weights='distance')
logistic = linear_model.LogisticRegression()
dt = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
ab = AdaBoostClassifier(dt, n_estimators=300)

rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
et = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)

models = [knn,logistic,dt,ab,rf,et]
for traincv, testcv in cv:
    for clf in models:
        clf.fit(train_df.ix[traincv,columns],train_df.ix[traincv,'Survived'])
    
        #predict the test set
        y_preds = clf.predict(train_df.ix[testcv,columns])
        y_test = train_df.ix[testcv,'Survived']
        accuracy = accuracy_score(y_test, y_pred)
        error_rate =  1-accuracy
        if error_rate<minimum['error']:
            minimum['error']=error_rate
            minimum['model']=model
    break

print 'final {}'.format(minimum)

