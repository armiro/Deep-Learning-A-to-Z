# Convolutional Neural Network Homework Assignment: What's That Pet?

import numpy as np
from keras.preprocessing import image
from keras.models import load_model


classifier = load_model(filepath='trained_model.h5')

img_address = 'dataset/single_prediction/cat_or_dog_2.jpg'

test_image = image.load_img(path=img_address, target_size=(64, 64))
test_image = image.img_to_array(img=test_image)
test_image = np.expand_dims(a=test_image, axis=0)

result = classifier.predict(x=test_image)

# prob = classifier.predict_proba(x=test_image, verbose=1)
# training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print('This image belongs to the "%s" class.' % prediction)
