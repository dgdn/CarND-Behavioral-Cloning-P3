import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

def load_data(path_prefix):
  lines = []
  with open(path_prefix+'/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  measurements = []
  images = []
  for line in lines:
    measurement_center = float(line[3])
    correction = 0.2
    measurement_left = measurement_center + correction
    measurement_right = measurement_center - correction

    img_center_path = path_prefix+'/IMG/' + line[0].split('/')[-1]
    img_left_path = path_prefix+'/IMG/' + line[1].split('/')[-1]
    img_right_path = path_prefix+'/IMG/' + line[2].split('/')[-1]
    img_center = cv2.resize(cv2.imread(img_center_path)[:,:,::-1], (320, 160))
    #img_left = cv2.imread(img_left_path)
    #img_right = cv2.imread(img_right_path)
    
    measurements.extend([measurement_center])
    images.extend([img_center]) 
  return (images, measurements)

def load_mul_data(paths):
  x, y = [],[]
  for path in paths:
    imgs, mes = load_data(path) 
    x = x + imgs
    y = y + mes
  return (np.array(x), np.array(y))

X_train, y_train = load_mul_data(['normal'])

X_train, X_valid, y_train , y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

X_train_reverse, y_train_reverse = load_mul_data(['reverse'])
X_train_recovery, y_train_recovery = load_mul_data(['recovery'])
X_train_curve, y_train_curve = load_mul_data(['curve'])

X_train = np.concatenate((X_train,X_train_reverse, X_train_recovery, X_train_curve))
y_train = np.concatenate((y_train,y_train_reverse, y_train_recovery, y_train_curve))

augmented_images = []
augmented_measurements = []
for image, measurement in zip(X_train, y_train):
  augmented_images.append(image)
  augmented_measurements.append(measurement)
  augmented_images.append(cv2.flip(image,1))
  augmented_measurements.append(measurement*-1.0) 

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

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
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), shuffle=True, nb_epoch=30)
model.save('model.h5')
     


