# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ARAVIND.P
RegisterNumber:  212224240015
*/
```
```py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

# Load the dataset
data = pd.read_csv("Salary.csv")

# Show first 5 rows
print(data.head())

# Check data types
print(data.info())

# Check for null values
print(data.isnull().sum())

# Encode the categorical column 'Position'
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

# Features and target
x = data[["Position", "Level"]]
y = data[["Salary"]]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the Decision Tree Regressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Predict using the test set
y_pred = dt.predict(x_test)

# Evaluate model performance
mse = metrics.mean_squared_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Predict salary for a new employee with Position encoded as 5 and Level 6
predicted_salary = dt.predict([[5, 6]])
print("Predicted Salary for Position 5 and Level 6:", predicted_salary)


```
## Output:

### DATA HEAD:

<img width="1100" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/2a04d1f8-db54-4611-a509-46d77461e25e">

### DATA INFO:

<img width="1100" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/db4c1397-3fb3-442b-b4d3-8d51da62aa07">

### ISNULL() AND SUM():

<img width="1090" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/12c831d0-592e-4eaa-8015-99bd6f7b625f">

### DATA HEAD FOR SALARY:

<img width="1100" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/7f5cc54c-0c4d-4ba5-9b38-65ade5e28a01">

### MEAN SQUARED ERROR:

<img width="1100" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/98f67052-b838-4313-bffa-6669d988205e">

### R2 VALUE:

<img width="1090" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/9e48bef7-60e4-40a5-881c-6edd2e47a138">

### DATA PREDICTION:

<img width="1116" alt="image" src="https://github.com/gauthamkrishna7/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/141175025/0e29307f-7447-42ac-89ba-df9feffcafc2">


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
