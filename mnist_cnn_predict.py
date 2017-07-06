import os.path
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K

testX = pd.read_csv('test.csv').values.astype('float32')
testX /= 255

img_rows, img_cols = 28, 28

testX = testX.reshape(testX.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)
num_classes = 10

model = Sequential()
model.add(Conv2D(32,
                 data_format='channels_last',
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_file = 'mnist-model2.hdf5'
model.load_weights(model_file)

testY = model.predict_classes(testX, verbose=2)

pd.DataFrame({"ImageId": list(range(1,len(testY)+1)),
              "Label": testY}).to_csv('submission2.csv', index=False, header=True)
