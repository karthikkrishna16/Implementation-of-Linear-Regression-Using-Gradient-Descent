# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1:Start


STEP 2:Import the required library and read the dataframe.


STEP 3:Write a function computeCost to generate the cost function.


STEP 4:Perform iterations og gradient steps with learning rate.


STEP 5:Plot the Cost function using Gradient Descent and generate the required graph.


STEP 6:End
## Program:
```
# Program to implement the linear regression using gradient descent.
# Developed by: TH KARTHIK KRISHNA
# RegisterNumber: 212223240067

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        #Calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv")
data.head()

#Assuming rhe last column is your target variable 'y' and the preceding columns.
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)

scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target calue for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")
```

## Output:
![Screenshot 2025-03-05 093733](https://github.com/user-attachments/assets/f465847e-c25a-47ec-9def-dc7615a672eb)


![Screenshot 2025-03-05 093827](https://github.com/user-attachments/assets/31f08e7c-b249-4a1e-a51d-b517f11227b7)


![Screenshot 2025-03-05 093855](https://github.com/user-attachments/assets/e09e3fd9-c29b-4d63-83cb-2a77c68009c5)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
