import os.path
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

batch_size = 128
num_classes = 10
epochs = 12

train = pd.read_csv('train.csv').values
trainY = np_utils.to_categorical(train[:,0].astype('int32'), num_classes)
trainX = train[:, 1:].astype('float32')
trainX /= 255

img_rows, img_cols = 28, 28

trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols, 1)

model = Sequential()
model.add(Conv2D(32, kernel_size(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          )
