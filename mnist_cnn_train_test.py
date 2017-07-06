import os.path
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import EarlyStopping

batch_size = 128
num_classes = 10
epochs = 30
model_file_name="mnist-model.hdf5"

train = pd.read_csv('train.csv').values
trainY = np_utils.to_categorical(train[:,0].astype('int32'), num_classes)
trainX = train[:, 1:].astype('float32')
trainX /= 255


img_rows, img_cols = 28, 28

trainX = trainX.reshape(trainX.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)


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


# Define early stopping monitor
early_stopping_monitor = EarlyStopping(patience=2)


model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
	  callbacks=[early_stopping_monitor]
          )

model.save(model_file_name)

score = model.evaluate(trainX, trainY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
