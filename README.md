# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import pandas, numpy and sklearn.                  
Calculate the values for the training data set.              
Calculate the values for the test data set.
Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DHARMALINGAM S
RegisterNumber:  212223040037
*/
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print("MSE =",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)

## Output:

![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot 2024-03-02 160543](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/1a6db55d-c311-4638-8ed0-8eeaf9d57924)

![Screenshot 2024-03-02 155658](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/33997dc8-98a3-4d8e-ae1d-6552375cb43f)

![Screenshot 2024-03-02 155654](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/a2f22643-3f03-47af-921d-169fabe1a6cd)

![Screenshot 2024-03-02 160150](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/ce3d4fba-366e-4848-b669-9192b30d660d)

![Screenshot 2024-03-02 160841](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/ceca4def-f2bc-41b9-8d8e-9747cdd0acbc)

![Screenshot 2024-03-02 160354](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/741ede3e-c837-43ff-872c-91490c22b6c1)

![ai ex2](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/8b5cc377-2d9e-41dd-904e-e938f0845886)

![Screenshot 2024-03-02 160936](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/d6d90d60-dfb9-44ec-978b-644674acdb22)

![Screenshot 2024-03-02 160957](https://github.com/AkilaMohan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/152275002/763e51d9-3222-4b5c-9935-46718ffa470a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
