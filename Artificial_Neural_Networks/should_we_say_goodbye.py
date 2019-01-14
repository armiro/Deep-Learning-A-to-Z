# Artificial Neural Network Homework Assignment: Should we say goodbye to that customer

# importing libraries
import numpy as np
import pickle
from keras.models import load_model

# loading the saved model (trained in ann.py)
classifier = load_model(filepath='trained_model.h5')

# loading StandardScaler class object fitted into training data (saved in ann.py)
with open(file='sc.txt', mode='rb') as input_file:
    sc = pickle.load(file=input_file)

# Predicting a single new observation

"""
Predict if the customer with the following information will leave the bank:
Geography: France, Credit Score: 600, Gender: Male, Age: 40, Tenure: 3, Balance: 60000,
Number of Products: 2, Has Credit Card: Yes, Is Active Member: Yes, Estimated Salary: 50000
"""
# predict the probability of exiting the bank
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print('There is a %.2f percent chance that this customer exits the bank;' % (new_prediction * 100))

# should we say goodbye to this customer?
new_prediction = int(new_prediction > 0.5)
print('Thus, this customer will probably %s the bank.' % ((1 - new_prediction) * 'stay in' or new_prediction * 'leave'))
