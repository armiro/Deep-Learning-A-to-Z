# Convolutional Neural Network Homework Assignment: What's That Pet?

import numpy as np
from keras.preprocessing import image
from keras.models import load_model


classifier = load_model(filepath='trained_model.h5')

img_address = 'dataset/single_prediction/cat_or_dog_2.jpg'

test_image = image.load_img(path=img_address, target_size=(64, 64), color_mode='rgb')
# print(test_image.shape)  # it will raise an error because "test_image" is not an array
test_image = image.img_to_array(img=test_image)
# print(test_image.shape)  # it will print (64, 64, 3) which means an rgb 64 by 64 image
test_image = np.expand_dims(a=test_image, axis=0)
# print(test_image.shape)  # it will print (1, 64, 64, 3)
result = classifier.predict(x=test_image)

# in keras, "predict_proba" is the same as "predict" attribute,
# seems that there is no way to get the probability value of each prediction
# prob = classifier.predict_proba(x=test_image, verbose=1)
# training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print('This image belongs to the "%s" class.' % prediction)
