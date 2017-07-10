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
from keras.callbacks import TensorBoard

batch_size = 128
num_classes = 10
epochs = 10

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
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


# Define early stopping monitor
callbacks = [EarlyStopping(monitor="loss",
		           min_delta=0,
		           patience=3, 
                           verbose=0,
                           mode='auto'),
	     TensorBoard(log_dir='./logs',
			 write_graph=True,
			 write_images=True,
			 embeddings_freq=2,
			 embeddings_layer_names=['1','2','3','4','5'],
			 
			)
	    ]

model.fit(trainX, trainY,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
	  callbacks=callbacks
	 )

model.save(model_file_name)

score = model.evaluate(trainX, trainY, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
