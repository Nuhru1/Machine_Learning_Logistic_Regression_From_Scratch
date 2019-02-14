# Machine_Learning_Logistic_Regression_From_Scratch

 Linear regression algorithms are used to predict/forecast values but logistic regression is used for classification tasks.
 In this repo I will use it to perform classification task on iris dataset. The iris dataset contain 3 classes but I will drop one class and perform the classification on the 2 classes.
 
# Sigmoid Function (Logistic Function)

Logistic regression algorithm also uses a linear equation with independent predictors to predict a value. The predicted value can be anywhere between negative infinity to positive infinity. We need the output of the algorithm to be class variable, i.e 0-no, 1-yes. Therefore, we are squashing the output of the linear equation into a range of [0,1]. To squash the predicted value between 0 and 1, we use the sigmoid function.

![sigmoid](https://user-images.githubusercontent.com/44145876/52537240-50ca5c80-2d9f-11e9-8bc5-b8b87cda32fe.png)  ![z](https://user-images.githubusercontent.com/44145876/52537261-925b0780-2d9f-11e9-955e-bad60b97bc69.png)


Z is the Linear Equation and g is the sigmoid function


![h](https://user-images.githubusercontent.com/44145876/52537268-c0d8e280-2d9f-11e9-820e-3cde80c4673b.png)

h is the Squashed output

We take the output(z) of the linear equation and give to the function g(x) which returns a squashed value h, the value h will lie in the range of 0 to 1.


let’s visualize the graph of the sigmoid function for better understanding of squashing.

![ssg](https://user-images.githubusercontent.com/44145876/52537347-d4387d80-2da0-11e9-9457-e5daf71ee061.png)


# Cost Function

we use a logarithmic loss function to calculate the cost for misclassifying.


![cost](https://user-images.githubusercontent.com/44145876/52537363-f9c58700-2da0-11e9-8e50-926d69b8028e.png)
![c](https://user-images.githubusercontent.com/44145876/52537364-fc27e100-2da0-11e9-8dcc-6d6fd7c18f8f.png)


# Calculating Gradients

the gradient will be used to update our theta_0, theta_1, theta_2, ...  in other to minimize our cost function after each iteration 


![grd](https://user-images.githubusercontent.com/44145876/52537423-b3245c80-2da1-11e9-9037-562fd93dd3b7.png)

# result

I got an Accuracy of 1.0 (100%)

# graph cost function 

![screenshot 18](https://user-images.githubusercontent.com/44145876/52537447-f7aff800-2da1-11e9-9ffb-bfa89607efba.png)

