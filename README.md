# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. start the program
2. Import numpy as np
3. Plot the points
4. Initialize the program
5. End the program

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: JEEVANANDAM M
RegisterNumber:  212222220017
```
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("50_Startups.csv")
data.head()

# Plot data
plt.scatter(data['R&D Spend'], data['Profit'])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

# Feature scaling
data['R&D Spend'] = (data['R&D Spend'] - data['R&D Spend'].mean()) / data['R&D Spend'].std()
data['Profit'] = (data['Profit'] - data['Profit'].mean()) / data['Profit'].std()

# Define computeCost and gradientDescent functions
def computeCost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    square_err = (h - y) ** 2
    return 1 / (2 * m) * np.sum(square_err)

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = []
    for i in range(num_iters):
        predictions = X.dot(theta)
        error = X.transpose().dot(predictions - y)
        descent = alpha * (1 / m) * error
        theta -= descent
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Predict profit using the trained model
predicted_profit = X.dot(theta)

# Print the predicted profit values
print("Predicted profit values:")
for i in range(len(predicted_profit)):
    print("Predicted profit for R&D Spend ${}: ${}".format(data['R&D Spend'][i], predicted_profit[i][0]))

# Prepare data for computation
X = np.column_stack((np.ones(len(data)), data['R&D Spend']))
y = data['Profit'].values.reshape(-1, 1)
theta = np.zeros((2, 1))

# Compute initial cost
initial_cost = computeCost(X, y, theta)
print("Initial cost:", initial_cost)

# Perform gradient descent
alpha = 0.01
num_iters = 1500
theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

# Print optimized parameters
print("Optimized parameters:", theta)

# Plot cost function
plt.figure()
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\\theta)$")
plt.title("Cost function using Gradient Descent")

# Plot the linear fit
plt.figure()
plt.scatter(data['R&D Spend'], data['Profit'])
plt.plot(data['R&D Spend'], X.dot(theta), color='red')
plt.xlabel("R&D Spend")
plt.ylabel("Profit")
plt.title("Profit Prediction")

plt.show()


```
## Output:

![ML 3 1 scr 316213804-ac35b08e-a32a-4721-8993-06253ced866c](https://github.com/jeeva078/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147048597/3920a366-e134-4f3f-a55c-c65a6087f92b)
![ml 3 2 scr  316213824-7cb3d0f6-e4ec-42d3-bc95-104631278144](https://github.com/jeeva078/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147048597/abb91c52-3651-41a6-b1fa-a1ed9a85bda5)
![ml 3 3 scr   316213764-187f654a-b0e5-4cbb-b9ee-36b09d79ecf6](https://github.com/jeeva078/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147048597/26899cf9-5a3e-40e2-b387-1f2f787309df)
![ml 3 4 scr   316213767-4dfcd137-6b63-4b1c-a866-90098b026f8a](https://github.com/jeeva078/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147048597/158ed290-e9ff-4511-8450-310e8b2e9d40)

![ML 3 5SCR  316213777-3acc9ddd-3055-4756-a5cc-917ba33ff5af](https://github.com/jeeva078/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/147048597/c2860430-c60e-4805-9d9a-f241bfd119df)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
