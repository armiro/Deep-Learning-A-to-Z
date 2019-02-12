# Artificial Neural Network


# Part 1 - Data Preprocessing

# Importing the libraries
import pandas as pd
import warnings

# Ignore warnings
warnings.filterwarnings(action='ignore')

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 - Evaluating, Improving and Tuning the ANN

# Evaluating and Improving the ANN (Dropout Regularization to reduce over-fitting if needed)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_classifier(add_dropout=False, dropout_rate=0.1):
    classifier = Sequential()
    classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    if add_dropout:
        classifier.add(layer=Dropout(rate=dropout_rate))
    classifier.add(layer=Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(layer=Dropout(rate=0.1))
    classifier.add(layer=Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=200, add_dropout=True, dropout_rate=0.1)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=2)
# 'n_jobs' defines how many CPU cores are used for the task; '-1' means all the CPU cores are used.
print("the average accuracy score (bias) is:", accuracies.mean())
print("the std deviation of accuracy scores (variance) is:", accuracies.std())


# Tuning the ANN using grid search cross-validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_classifier(optimizer, add_dropout=False, dropout_rate=0.1):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    if add_dropout:
        classifier.add(layer=Dropout(rate=dropout_rate))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [32, 48], 'epochs': [500, 600], 'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print("best parameters are:", grid_search.best_params_)
print("best accuracy is:", grid_search.best_score_)

