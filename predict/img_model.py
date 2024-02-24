#from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential

img_width = 150
img_height = 150



def load_model(MODEL_2):
	model =Sequential()

	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(32,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(64,(3,3), input_shape=(img_width, img_height, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.load_weights(MODEL_2)
	return model

