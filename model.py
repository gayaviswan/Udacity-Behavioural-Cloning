import csv
import os
import cv2
from scipy import ndimage
import numpy as np
import sklearn
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt


"""
Flip the image based on a toss of a coin. 
Input:
image : Input Image
steering_angle: Steering angle
Output:
Output Image - Flipped or not based on the toss
Steering angle 
"""
def flip(image, steering_angle):
    head = np.random.binomial(1, 0.5)
    if head:
        image_flipped = np.fliplr(image)
        measurement_flipped = -steering_angle
        return image_flipped, measurement_flipped
    else:
        return image, steering_angle

        
"""
Generator that continuously generates batches of image
"""
def generator(batch_size=64):
    while True: # Loop forever so the generator never terminates
        
        X_train = []
        y_train = []
        
        data = pd.read_csv('/opt/data/driving_log.csv')
        num_img = len(data)
        indices = np.random.randint(0, num_img, batch_size)
        batch_images = []
        for index in indices:
            rnd_image = np.random.randint(0, 3)
            if rnd_image == 1:
                img = data.iloc[index]['center'].strip()
                angle = data.iloc[index]['steering']
                batch_images.append((img, angle))
            elif rnd_image == 0:
                img = data.iloc[index]['left'].strip()
                angle = data.iloc[index]['steering'] + 0.25
                batch_images.append((img, angle))
            elif rnd_image == 2:                                                        
                img = data.iloc[index]['right'].strip()
                angle = data.iloc[index]['steering'] - 0.25
                batch_images.append((img, angle))
        for img_file, angle in batch_images:
            name = '/opt/data/IMG/'+ img_file.split('/')[-1]
            unproc_img = plt.imread(name)
            unproc_ang = angle
            new_image = unproc_img
            top = int(np.ceil(new_image.shape[0] * 0.35))
            bottom = new_image.shape[0] - int(np.ceil(new_image.shape[0] * 0.12))
            new_image = new_image[top:bottom, :]
            rnd_flip = np.random.randint(0, 2)
            #rnd_flip = 1
            if rnd_flip == 1:
                new_image, new_angle = new_image, unproc_ang
            else:
                new_image, new_angle = flip(new_image, unproc_ang)
            new_image = cv2.resize(new_image, (64,64))
                
            X_train.append(new_image)
            y_train.append(new_angle)
        yield np.array(X_train), np.array(y_train)

        
# compile and train the model using the generator function
train_generator = generator()
validation_generator = generator()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam

# use NVIDIA pipeline
model = Sequential()


model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(64, 64, 3)))
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Flatten())

model.add(Dense(1164))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history_object = model.fit_generator(train_generator, samples_per_epoch=1032, validation_data=validation_generator, nb_val_samples=248, nb_epoch=10, verbose=1)

print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('modelAccuracy.png')

model.save('model.h5')

