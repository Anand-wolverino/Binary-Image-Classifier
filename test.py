# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:05:14 2019

@author: anand
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image


model = load_model('weights.h5')

# Load the test image by giving the correct path
test_image = image.load_img('', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result)

