#importing the dependencies 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

#loading data file - PIMA Dİibets data set 
diabetes_dataset = pd.read_csv('diabetes.csv')

# pd.read_csv? to see parameters of func.

#printing the first and last 5 rows
print(diabetes_dataset.head())
print(diabetes_dataset.tail())

#numbe of rows and coloumns
diabetes_dataset.shape

# getting statistical measures
diabetes_dataset.describe()

print(diabetes_dataset['Outcome'].value_counts())
#0 : non diabetic, 1: diabetic

print(diabetes_dataset.groupby('Outcome').mean())

#separating the data and labels 
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

#data standardization 
scaler = StandardScaler()
scaler.fit(X)
standardizied_data = scaler.transform(X)
print(standardizied_data)

X = standardizied_data
y = diabetes_dataset['Outcome']
print(X)
print(y)

#Train-Test
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.2 , stratify
=Y, random_state=2 )

print(X.shape, xtrain.shape, xtest.shape )

#training the model 
classifier = svm.SVC(kernel='linear')

#training the support vector machine classifier
classifier.fit(xtrain, ytrain)

#model evolution - accuracy score 
xtrain_prediction = classifier.predict(xtrain)
training_data_accuracy = accuracy_score(xtrain_prediction, ytrain)
print('accuracy score of training data: ',training_data_accuracy )

xtest_prediction = classifier.predict(xtest)
test_data_accuracy = accuracy_score(xtest_prediction, ytest)
print('accuracy score of ttest data: ',test_data_accuracy )

#making a predictive system 
input_data =(4,110,92,0,0,37.6,0.191,30)

#changeing input_data to numpy array
input_data_asnparray = np.asarray(input_data)

#reshape the array 
input_data_reshaped = input_data_asnparray.reshape(1,-1)

#standardizing the input data 
std_data = scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0]==0):
  print('the person is not diabetic')
else: 
  print('the person is diabetic')
  






