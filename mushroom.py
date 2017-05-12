import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.cross_validation import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data.csv")

labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

train, test = train_test_split(data, test_size = 0.3)

train_X = train[list(data.columns[1:23])]
train_Y = train['class']

test_X = test[list(data.columns[1:23])]
test_Y = test['class']

model = RandomForestClassifier(n_estimators=100)

model.fit(train_X, train_Y)

prediction = model.predict(test_X)



