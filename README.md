# Python-Machine-Learning
df.shape --- row count and colume count
df.describe() --- basic information
df.values

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = mdata.drop(columns = ['gen'])
y = mdata['gen']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2) --- 0.2 means 20% of data for testing

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
---- predictions = model.predict([[21,1],[21,2]])   ---- age 20, gender 1
-----predictions
predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)
score
