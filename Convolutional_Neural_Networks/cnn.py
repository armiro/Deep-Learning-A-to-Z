# Convolutional Neural Network

# Installing Tensorflow
# pip install tensorflow

# facing problem? install via wheel file url address (or download the file and install via pip using local file address)
# python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.12.0-py3-none-any.whl

# If you have problems running tf on windows for python v3.7 without Anaconda
# (or other versions which there is no official google release of tf for them), then download your desired
# wheel package from this github repo:
# https://github.com/fo40225/tensorflow-windows-wheel
# and then, install using pip. For example: pip install tensorflow-1.12.0-cp37-cp37m-win_amd64.whl
# That is the easiest possible solution, to the best of my knowledge.

# Installing Keras (needs Scipy and some other packages as dependencies)
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Click on a function, then use "ctrl + shift + i" to see more information about input arguments
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=None))

# Adding a second convolutional layer
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# devising two pooling layers beside each other, accuracy has been increased to some extent

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=32, activation='relu'))
# classifier.add(LeakyReLU(alpha=0.2))
# using 'LeakyReLU' did not end in a significant change
classifier.add(Dense(units=1, activation='sigmoid'))
# changing activation function to 'tanh' had worse result

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# using predefined 'SGD' had worse result, 'adamax' had the same result as 'adam'

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                   rotation_range=45)
# adding rotation_range resulted in a few increase, whereas adding 'vertical_flip' had worsen the result
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=16,
                                                 class_mode='binary')
# increasing 'batch_size' slows down the system, however a slight increase is observed
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=16,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set,
                         validation_steps=2000)


# save the classifier and weights as an HDF5 file
classifier.save('trained_model.h5')
