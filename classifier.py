# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 20:26:52 2018

@author: anand
"""



import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Building the sequential model Neural net
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the neural net with the loss function , optimiser and the metrics
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Using an image data generator to rescale , flip images to generate more images for training purposes
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)

# Using flow from directory to point to the directory containing images. Give your path to image direcroy here
training_set = train_datagen.flow_from_directory('D:\\lbp\\neural',target_size = (128,128),batch_size = 10,class_mode = 'binary')

# Fit the model on the training images
classifier.fit_generator(training_set,steps_per_epoch = 70,epochs = 5)

classifier.save('weights.h5')
# Finally ! Testing the classifier. Give your path to the test image here!
test_image = image.load_img('D:\\datasets\\bea1.jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(classifier.predict_proba(test_image))
print(result)