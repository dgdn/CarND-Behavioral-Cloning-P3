import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        # 0 track for normal image, 1 track for image to be flipped
        samples.append((0, line))
        samples.append((1, line))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample_data in batch_samples:
                flag, batch_sample = batch_sample_data
                    # normal
                if flag == 0:
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                else:
                    # flip image and angle
                    name = './data/IMG/'+batch_sample[0].split('/')[-1]
                    center_image = cv2.flip(cv2.imread(name), 1)
                    center_angle = float(batch_sample[3]) * -1.0
                    images.append(center_image)
                    angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 128

train_samples, validation_samples = train_test_split(samples, test_size=0.1)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
#model.add(BatchNormalization(input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, 
  samples_per_epoch=len(train_samples),
  validation_data=validation_generator,
  nb_val_samples=len(validation_samples), nb_epoch=30)
  
model.save('model.h5')
     


