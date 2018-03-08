import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

lines = []
with open('data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)

measurements = []
images = []
for line in lines:
  measurements.append(float(line[3]))
  img_path = 'data/IMG/' + line[0].split('/')[-1]
  images.append(cv2.imread(img_path))

X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape, y_train.shape)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Convolution2D(16, 5, 5, border_mode='valid'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(120))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(84))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)
model.save('model.h5')
     


