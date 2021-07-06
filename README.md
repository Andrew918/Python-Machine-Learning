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

-----------------------------------------------
A/B TESTING

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss
from statistics import mean

data = pd.read_csv('C:\\Users\\yhu\\OneDrive - OneWorkplace\\Documents\\BB Triggers Weekly\\\data.csv')

#sns.distplot(data.Conversion_A)
#sns.distplot(data.Conversion_B)

t_stat, p_val= ss.ttest_ind(data.Conversion_B,data.Conversion_A)

control = statistics.mean(data.Conversion_A)
test = statistics.mean(data.Conversion_B)

t_stat , p_val, control, test

# p_value 0.0003638 is less than significance level which is 0.05. Hence, we can reject the null hypothesis. 
# This means that in our A/B testing, newsletter B is performing better than newsletter A. 
# So our recommendation would be to replace our current newsletter with B to bring more traffic on our website.

----------------------------------------------------------
