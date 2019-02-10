import pandas as pd

dataset = pd.read_csv('iris.csv')
# our dataset have 3 classes 
#but we decided to drop one class and use 2 classes for our classification
data = dataset.iloc[ : 99 , :]

target = data.iloc[ : , -1: ]

y = []
# let's assign value 1 to class 'Iris-setosa' and 0 to 'Iris-versicolor'
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
# shuffling our data before splitting:
shuffle(x, y)

x_train = []
x_test = []
y_train = []
y_test = []
# splitting our data to train and test sets with 20% for test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)


x_1 = x_train[:,0]
x_2 = x_train[:,1]
x_3 = x_train[:,2]
x_4 = x_train[:,3]


x_1 = np.array(x_1)
x_2 = np.array(x_2)
x_3 = np.array(x_3)
x_4 = np.array(x_4)

x_1 = x_1.reshape(79,1)
x_2 = x_2.reshape(79,1)
x_3 = x_3.reshape(79,1)
x_4 = x_4.reshape(79,1)

y_train = y_train.reshape(79, 1)


# let's define the sigmoid function:
def sigmoid(x):
    return (1/( 1+ np.exp(-x)))



m = len(x_train)
alpha = 0.0001

theta_0 = np.zeros((m, 1))
theta_1 = np.zeros((m ,1))
theta_2 = np.zeros((m ,1))
theta_3 = np.zeros((m ,1))
theta_4 = np.zeros((m ,1))


epoch = 0
cost_func = []

while epoch < 100000:
    
    y = theta_0 + theta_1 * x_1 + theta_2 * x_2 + theta_3 * x_3 + theta_4 * x_4
    
    # passing the output of the linear equation to the sigmoid function for squaching the value to [0,1]
    y = sigmoid(y)
    
    # the cost function (not an array, but a value which have to be minimize by updating theta after each iteration)
    cost = (- np.dot(np.transpose(y_train), np.log(y)) - np.dot(np.transpose(1-y_train), np.log(1-y)))/m
    
    # let's get the gradients of our 4 theta in other to update our thetas 
    # let's define x_0 as : np.ones((1, m))
    theta_0_grad = np.dot(np.ones((1, m)) , y-y_train)/m
    theta_1_grad = np.dot(np.transpose(x_1), y-y_train)/m
    theta_2_grad = np.dot(np.transpose(x_2), y-y_train)/m
    theta_3_grad = np.dot(np.transpose(x_3), y-y_train)/m
    theta_4_grad = np.dot(np.transpose(x_4), y-y_train)/m
    
    # let's update thetas now:
    
    theta_0 = theta_0 - alpha * theta_0_grad
    theta_1 = theta_1 - alpha * theta_1_grad
    theta_2 = theta_2 - alpha * theta_2_grad
    theta_3 = theta_3 - alpha * theta_3_grad
    theta_4 = theta_4 - alpha * theta_4_grad
                 
    
    cost_func.append(cost)
    epoch+=1
    

    
# let's test with our test data and see the result  
    
from sklearn.metrics import accuracy_score


test_x_1 = x_test[:,0]
test_x_2 = x_test[:,1]
test_x_3 = x_test[:,2]
test_x_4 = x_test[:,3]

test_x_1 = np.array(test_x_1)
test_x_2 = np.array(test_x_2)
test_x_3 = np.array(test_x_3)
test_x_4 = np.array(test_x_4)

test_x_1 = test_x_1.reshape(20,1)
test_x_2 = test_x_2.reshape(20,1)
test_x_3 = test_x_3.reshape(20,1)
test_x_4 = test_x_4.reshape(20,1)

index = list(range(20,89))

theta_0 = np.delete(theta_0, index)
theta_1 = np.delete(theta_1, index)
theta_2 = np.delete(theta_2, index)
theta_3 = np.delete(theta_3, index)
theta_4 = np.delete(theta_4, index)

theta_0 = theta_0.reshape(20, 1)
theta_1 = theta_1.reshape(20, 1)
theta_2 = theta_2.reshape(20, 1)
theta_3 = theta_3.reshape(20, 1)
theta_4 = theta_4.reshape(20, 1)

y_pred = theta_0 + theta_1 * test_x_1 + theta_2 * test_x_2+ theta_3 * test_x_3+ theta_4 * test_x_4
y_pred = sigmoid(y_pred)

final_y_pred = []
# here we set our threshold to 0.5. 
# all values greater than it will return 1 and less will return 0
for val in y_pred:
    if val >= 0.5:
        final_y_pred.append(1)
    else:
        final_y_pred.append(0)

# printing our accuracy       
print(accuracy_score(y_test,final_y_pred))


# plotting our cost function
import matplotlib.pyplot as plt 

cost_func = np.array(cost_func)
cost_func = cost_func.reshape(100000,1)
plt.plot(range(len(cost_func)),cost_func)