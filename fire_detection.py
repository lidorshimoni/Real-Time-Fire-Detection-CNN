from __future__ import print_function
import keras.layers
import keras
from keras.datasets import cifar10
import os
from keras.layers import Dense, Dropout, Activation, Flatten
# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# create generator
datagen = ImageDataGenerator(rescale=1/255,
                             featurewise_center=False,  # set input mean to 0 over the dataset
                             samplewise_center=False,  # set each sample mean to 0
                             featurewise_std_normalization=False,  # divide inputs by std of the dataset
                             samplewise_std_normalization=False,  # divide each input by its std
                             zca_whitening=False,  # apply ZCA whitening
                             zca_epsilon=1e-06,  # epsilon for ZCA whitening
                             rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                             # randomly shift images horizontally (fraction of total width)
                             width_shift_range=0.1,
                             # randomly shift images vertically (fraction of total height)
                             height_shift_range=0.1,
                             shear_range=0.,  # set range for random shear
                             zoom_range=0.,  # set range for random zoom
                             channel_shift_range=0.,  # set range for random channel shifts
                             # set mode for filling points outside the input boundaries
                             fill_mode='nearest',
                             cval=0.,  # value used for fill_mode = "constant"
                             horizontal_flip=True,  # randomly flip images
                             vertical_flip=False,  # randomly flip images
                             # set function that will be applied on each input
                             preprocessing_function=None,
                             # image data format, either "channels_first" or "channels_last"
                             data_format=None,
                             # fraction of images reserved for validation (strictly between 0 and 1)
                             validation_split=0.0)


batch_size = 10
num_classes = 3
epochs = 50
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Fire_detection_vgg16_trained_model.h5'

data_dir = save_dir = os.path.join(os.getcwd(), 'FIRE-SMOKE-DATASET')


X_train = datagen.flow_from_directory(data_dir + '/Train/',
                                    class_mode='categorical',
                                    batch_size=batch_size,
                                    target_size=(150, 150)
                                    )

# load and iterate test dataset
X_test = datagen.flow_from_directory(data_dir + '/Test/',
                                   class_mode='categorical',
                                   batch_size=batch_size,
                                   target_size=(150, 150)
                                   )


base_model = keras.applications.vgg16.VGG16(include_top=False,
                                            weights='imagenet',
                                            input_tensor=None,
                                            input_shape=(150, 150, 3),
                                            pooling=None, classes=num_classes)
top_model = base_model.output
#
top_model = (Flatten())(top_model)
top_model = (Dense(256, activation='relu'))(top_model)

# top_model = Dropout(0.1)(top_model)
top_model = (Dense(num_classes, activation='softmax'))(top_model)

model = keras.models.Model(inputs=base_model.input, outputs=top_model)

for layer in base_model.layers:
    layer.trainable = False


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])




def train():
    model.fit_generator(generator=X_train,
                        validation_data=X_test,
                        epochs=epochs,
                        steps_per_epoch=100,
                        validation_steps=100
    )



def save():
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def score():
    # Score trained model.
    scores = model.evaluate_generator(generator=X_test, steps=20)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def learn():
    train()
    save()
    score()
