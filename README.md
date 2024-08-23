# EX-01 Developing a Neural Network Regression Model
## AIM

To develop a neural network regression model for the given dataset.

## THEORY

1) Neural networks consist of simple input/output units called neurons (inspired by neurons of the human brain). These input/output units are interconnected and each connection has a weight associated with it.

2) Regression helps in establishing a relationship between a dependent variable and one or more independent variables. Regression models work well only when the regression equation is a good fit for the data. Most regression models will not fit the data perfectly.

3) First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 3 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model
![image](https://github.com/user-attachments/assets/1ed70f24-ef2d-4c4a-b2bc-e7369249edef)

## DESIGN STEPS
### STEP 1:Loading the dataset
### STEP 2:Split the dataset into training and testing
### STEP 3:Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:Build the Neural Network Model and compile the model.
### STEP 5:Train the model with the training data.
### STEP 6:Plot the performance plot
### STEP 7:Evaluate the model with the testing data.
## PROGRAM
```
DEVELOPED BY : MUKIL KUMAR V
REGISTER NUMBER : 212222230087
```

## Importing Required packages
```py
from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd
```

## Authenticate the Google sheet
```py
auth.authenticate_user()
creds,_=default()
gc=gspread.authorize(creds)
worksheet = gc.open('experiment1').sheet1
data = worksheet.get_all_values()
```
## Construct Data frame using Rows and columns
```py
df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'input':'float'})
df = df.astype({'output':'float'})
df.head()
x=df[['input']].values
y=df[['output']].values
x
```
## Split the testing and training data
```py
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1=Scaler.transform(X_train)
```
## Build the Deep learning Model
```py
ai_brain=Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(10,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train.astype(np.float32),epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
```

## Evaluate the Model
```py
X_test1=Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1=[[5]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
```
## Dataset Information
![Screenshot 2024-08-23 092904](https://github.com/user-attachments/assets/c17e2446-84f8-4624-a323-843de63aaa3e)

## Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/ccb72680-7254-4ca2-b7bc-603e4591bcee)
## Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/c8b6d8fe-de28-4579-86c1-5445434b5c30)
![image](https://github.com/user-attachments/assets/040d8eae-ce17-4d26-8f2c-9f9900a7e984)
## New Sample Data Prediction
![image](https://github.com/user-attachments/assets/d6a4dfb8-8ace-44e2-9248-2d07d0396160)

## RESULT
Thus a Neural network for Regression model is Implemented
