# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:14:40 2020

@author: reine
"""
# Functions
def key_performance_indicators(confusion_matrix):
    cm = confusion_matrix
    print(cm)
    
    # Total accuracy and misclassification calculations
    total_right = cm[0][0] + cm[1][1]
    total_wrong = cm[0][1] + cm[1][0]
    total_acc = total_right / (total_right + total_wrong)
    print(f'The total accuracy of the Model is {total_acc:.3f}')
    
    misclass = total_wrong / (total_right + total_wrong)
    print(f'The total misclassification rate (error) of the Model is {misclass:.3f}')
    
    # TP, FP, FN - Precision, Recall, Specificity, F1 calculations
    tp = cm[0][0]
    fp = cm[1][0]
    fn = cm[0][1]
    tn = cm[1][1]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    
    # Print KPI
    print(f'The precision of the Model is {precision:.3f}')
    print(f'The recall of the Model is {recall:.3f}')
    print(f'The specificity of the Model is {specificity:.3f}')
    print(f'The F1 Score of the Model is {f1:.3f}')

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import keras
from keras.models import Sequential
from keras.layers import Dense

# Load CSV
data = pd.read_csv('adult-income.csv')

# Check for Null values
print(data.isnull().sum(axis = 0))
print(data.dtypes)

# Create DataFrame and One Hot Encode
#df = pd.DataFrame(data)
df = pd.get_dummies(data, drop_first=True)
print(f'DataFrame Shape: {df.shape}')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, stratify=y)

# Decision Tree
print('running Decision Tree...')

# Train the model
dtclassifier = DecisionTreeClassifier(random_state=42)
dtclassifier.fit(X_train, y_train)

# Make prediction
y_predict = dtclassifier.predict(X_test)

# Model Evaluation
cm_dt = confusion_matrix(y_test, y_predict)

print('---Decision Tree---')
key_performance_indicators(cm_dt)

# Random Forest
print('\n running Random Forest Classifier...')

# Train the model
rfclassifier = RandomForestClassifier(random_state=42)
rfclassifier.fit(X_train, y_train)

# Make prediction
y_predict = rfclassifier.predict(X_test)

# Evaluate the Model
cm_rfc = confusion_matrix(y_test, y_predict)

print('---Random Forest Classifier')
key_performance_indicators(cm_rfc)

# ANN
print('\n running ANN...')

# Build the model
num_neurons = 28
num_layers = 9
num_epochs = 200
num_batch = 5
act_func = 'relu'

model = Sequential()
model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func, input_dim = 19)) # Input layer & 1 hidden layer

i = 0

while i < num_layers-1:
    model.add(Dense(units = num_neurons, kernel_initializer = 'uniform', activation = act_func))
    i += 1

model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # Output layer with 3 outputs

# Compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=num_epochs, batch_size=num_batch)

# Make prediction
y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)

# Evaluate the Model
cm_ann = confusion_matrix(y_test, y_pred)

print('---ANN---')
key_performance_indicators(cm_ann)

# SVM
print('\n running SVM...')

# Train the model
svc = SVC(kernel='poly', C = 0.5, degree = 2, gamma = 3.0)
svc.fit(X_train, y_train)

# Make prediction
y_predict = svc.predict(X_test)
#y_pred = (y_pred > 0.5)
# Evaluate the Model
cm_svm = confusion_matrix(y_test, y_predict)
print('---SVM---')
key_performance_indicators(cm_svm)

print('\n KEY PERFORMANCE INDICATORS')
print('---Decision Tree---')
key_performance_indicators(cm_dt)
print('\n---Random Forest Classifier')
key_performance_indicators(cm_rfc)
print('\n---ANN---')
key_performance_indicators(cm_ann)
print('\n---SVM---')
key_performance_indicators(cm_svm)







