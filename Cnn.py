

import pandas as pd
import numpy as np
import glob
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Flatten, Dense, Dropout, MaxPooling2D

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator


from numpy.random import seed
seed(1337)
from tensorflow.random import set_seed
set_seed(1337)


import warnings
warnings.filterwarnings('ignore')



print(tf.__version__)

def load_samples_as_images(pathX, pathY,img_width, img_height):
    files = sorted(glob.glob(pathX))
    labels_df = pd.read_csv(pathY)
    Y = np.array(labels_df[' hemorrhage'].tolist())
    images = np.empty((len(files), img_width, img_height))
    for i, _file in enumerate(files):
        images[i, :, :] = cv2.resize(cv2.imread(_file, 0), (img_width, img_height))
    return images, Y





def cnnModel(img_width, img_height, pathX, pathY):
    images, Y = load_samples_as_images(pathX, pathY, img_width, img_height)
    train_images, test_images, train_labels, test_labels = train_test_split(images, Y, test_size=0.2,
                                                                            random_state=1)
    val_images, test_images, val_labels, test_labels = train_test_split(test_images, test_labels, test_size=0.5,
                                                                        random_state=1)
    


    model = Sequential()
   


    model.add(tf.keras.layers.Input(shape=(img_height,img_width,1)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    


    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())



    model.add(Dense(64))
    model.add(Activation('relu'))



    model.add(Dropout(0.5))



    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    
    
    model.compile(loss='binary_crossentropy',
                                optimizer='rmsprop',
                                metrics=['accuracy'])
    
   

    nb_train_samples = len(train_images)
    nb_validation_samples = len(val_images)
    epochs = 100
    batch_size = 10


    
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.0,
        zoom_range=0.1,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)




    train_generator = train_datagen.flow(
        train_images[..., np.newaxis],
        train_labels,
        batch_size=batch_size)

    validation_generator = val_datagen.flow(
        val_images[..., np.newaxis],
        val_labels,
        batch_size=batch_size)




    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)



    print("Final accuracy: " + str(model.evaluate(test_images[..., np.newaxis] / 255., test_labels)[1] * 100) + "%")


    model.save('D:\\Head-CT-hemorrhage-detection-master\\Code\\NAKSH.model.keras')
    
    