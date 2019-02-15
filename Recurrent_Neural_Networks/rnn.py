# Recurrent Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
# we need to use "1:2" instead of "1" in order to have a numpy array instead of a vector.
# and then, using "values" attribute we convert it to the numpy array

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# we have two feature scaling methods: Standardization(StandardScaler) and Normalization (MinMaxScaler)
# rule of thumb: when having "sigmoid" as the activ. func., better to use Normalization (like here)

# Creating a data structure with 60 timesteps and 1 output
X_train = list()
y_train = list()
for i in range(60, len(training_set)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
# based on the stock price in the last 60 days, we are going to predict it on the next day
# hence, timestep is equal to 60. notice that len(training_set) is 1258.

# Reshaping
X_train = np.reshape(a=X_train, newshape=(X_train.shape[0], X_train.shape[1], 1))
# for using as the training-set of the RNN models, we need to reshape the input
# the newshape is: (batch_size or num_of_records, num_of_timesteps, num_of_indicators)
# num_of_indicators is equal to 1, because we only use "Open" as the input feature


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.2))
# the "units" means the number of neurons to add in the hidden layer in each LSTM layer
# using dropout rate equal to 0.2 means that here 10 neurons are ignored


# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(rate=0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50))
regressor.add(Dropout(rate=0.2))
# in the last LSTM layer, the 'return_sequences' should be 'False' which is the default value of it

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')
# usually, 'rmsprop' is the best optimizer to choose when working with RNNs

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# axis=0 means vertically (columns), axis=1 means horizontally (rows)

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)

X_test = list()
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# we need to do inverse scaling because the model was fitted on the scaled y_train (line 29)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time'), plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
