#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 13:38:08 2019

@author: magshimim
"""


from __future__ import print_function
import keras.layers
import keras
from keras.datasets import cifar10
import os
from keras.layers import Dense, Dropout, Activation, Flatten
# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt

import numpy as np
# create generator
datagen = ImageDataGenerator(rescale=1/255)

batch_size = 10
num_classes = 3
epochs = 50
data_augmentation = False
num_predictions = 20
main_dir = r"C:\Users\king\Desktop\2DO\Real-Time-Fire-Detection-CNN-master/"
save_dir = main_dir + "Trained Models/"
model_name = 'Fire_detection_vgg16_trained_model.h5'
data_dir = main_dir + 'FIRE-SMOKE-DATASET'


# X_train = datagen.flow_from_directory(data_dir + '/Train/',
#                                     class_mode='categorical',
#                                     batch_size=batch_size,
#                                     target_size=(150, 150),
#                                     shuffle=True
#                                     )
#
# # load and iterate test dataset
# X_test = datagen.flow_from_directory(data_dir + '/Test/',
#                                    class_mode='categorical',
#                                    batch_size=batch_size,
#                                    target_size=(150, 150),
#                                     shuffle=True
#                                    )

#Trained Models
model = load_model(save_dir + model_name)
for layer in model.layers:
    layer.trainable = False


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def save():
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

#
# def score():
#     # Score trained model.
#     scores = model.evaluate_generator(generator=X_test, steps=20)
#     print('Test loss:', scores[0])
#     print('Test accuracy:', scores[1])

img = np.ndarray(shape=(4,150,150,3))

for i in range(1,4):
    img[i] = plt.imread(main_dir + 'test' + str(i) +'.jpg')
    print(model.predict(img)[i].argmax())

from keras.utils import plot_model
plot_model(model, to_file='model.png')
#0=fire 1=neutral 2=smoke