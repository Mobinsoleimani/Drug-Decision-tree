import numpy as np 
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("drug.csv")
# print(df)

x = df [['Age','Sex','BP','Cholesterol','Na_to_K']].values

y = df [['Drug']].values

##chang str to int in dataset
sex = preprocessing.LabelEncoder()
sex.fit(['F','M'])
x[:,1] = sex.transform(x[:,1]) 


bp = preprocessing.LabelEncoder()
bp.fit(['HIGH','LOW','NORMAL'])
x[:,2] = bp.transform(x[:,2])


ch = preprocessing.LabelEncoder()
ch.fit(['HIGH','NORMAL'])
x[:,3] = ch.transform(x[:,3])

xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size=0.3, random_state=3)

model = DecisionTreeClassifier(criterion='entropy',max_depth=4)
model.fit(xtrain, ytrain)
ptree = model.predict(xtest)
print(ptree)
print(ytest)
acc = metrics.accuracy_score(ytest, ptree)
print('scurecy score',acc)