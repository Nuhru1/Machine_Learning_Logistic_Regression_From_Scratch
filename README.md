# Machine_Learning_Logistic_Regression_From_Scratch

 Linear regression algorithms are used to predict/forecast values but logistic regression is used for classification tasks.
 In this repo I will use it to perform classification task on iris dataset. The iris dataset contain 3 classes but I will drop one class and perform the classification on the 2 classes.
 
# Sigmoid Function (Logistic Function)

Logistic regression algorithm also uses a linear equation with independent predictors to predict a value. The predicted value can be anywhere between negative infinity to positive infinity. We need the output of the algorithm to be class variable, i.e 0-no, 1-yes. Therefore, we are squashing the output of the linear equation into a range of [0,1]. To squash the predicted value between 0 and 1, we use the sigmoid function.

![sigmoid](https://user-images.githubusercontent.com/44145876/52537240-50ca5c80-2d9f-11e9-8bc5-b8b87cda32fe.png)  ![z](https://user-images.githubusercontent.com/44145876/52537261-925b0780-2d9f-11e9-955e-bad60b97bc69.png)


