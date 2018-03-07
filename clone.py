import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

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
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)
model.save('model.h5')
     


