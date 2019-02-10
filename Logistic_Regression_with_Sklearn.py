import pandas as pd

dataset = pd.read_csv('iris.csv')

data = dataset.iloc[ : 99 , :]

target = data.iloc[ : , -1: ]

y = []

for x in target.values:
    if x == 'Iris-setosa':
        y.append(1)
    else:
        y.append(0)

x = data.iloc[ : , : -1]
x = x.values.tolist()



from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

shuffle(x, y)

x_train = []
x_test = []
y_train = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test,y_pred))