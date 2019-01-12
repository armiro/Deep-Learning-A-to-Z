# Artificial Neural Network

# Installing Theano, Tensorflow and Keras is explained within the cnn.py file
# in the Convolutional_Neural_networks folder

# Part 1 - Data Preprocessing

# Importing the libraries
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option("display.max_columns", 14)

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
print(dataset.head())

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(y=X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(y=X[:, 2])
# the 'LabelEncoder' encodes categorical variables into numerical ones
# (e.g. France/Germany/Spain --> 2/1/0)

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X=X).toarray()
# the 'OneHotEncoder' encodes labeled categorical variables into separate binary columns
# (e.g. France/Germany/Spain --> 2/1/0 --> 100/010/001)
# the result will be placed as the first columns (now, we have 3 columns instead of 1)

X = X[:, 1:]
# we omit the first column in order to avoid dummy variable trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X=X_train)
X_test = sc.transform(X=X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
# 'input_dim' is the number of features, 'output_dim' should be found experimentally; however,
# a rule of thumb is to average the number of nodes in input layer and output layer (11+1/2.=6.)

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# for multi-class classification, change the 'units' based on the OneHotEncoded output variable
# and, change 'activation' to 'softmax'

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# for multi-class classification, set 'loss' as 'categorical_crossentropy'

# Fitting the ANN to the Training set
classifier.fit(x=X_train, y=y_train, batch_size=10, epochs=100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(x=X_test)
# to use confusion_matrix, we have to binarize the sigmoid's probable output
y_pred = (y_pred > 0.5)
# this means if y_pred > 0.5: y_pred is True, else: y_pred is False

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

TN = cm[0][0]; FP = cm[0][1]; FN = cm[1][0]; TP = cm[1][1]

acc = (TN + TP) / (TN + FP + FN + TP)
# accuracy is the number of all true classified samples divided by total number of samples

print('accuracy on the test-set is:', acc)

