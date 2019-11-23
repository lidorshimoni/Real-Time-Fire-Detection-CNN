from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib
import numpy as np
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
set = np.ndarray(shape=(900, 3, 150, 150))

for i in range(900):
    img = load_img('FIRE-SMOKE-DATASET/Train/Smoke/image_' + i + '.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    # x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    set += x


def generate(set, dir):
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(set, batch_size=1,
                              save_to_dir='augmented' + '/' + dir, save_format='jpg'):
        i += 1
        if i >= 3:
            break  # otherwise the generator would loop indefinitely

