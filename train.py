######   BUILD, TRAIN AND SAVE THE CONVOLUTIONAL MODEL    ########


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Activation
from keras.layers import Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
#####  LOADING AND EXTRACTING DATA   #####


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle


def data_loader():
    # Load dataset file
    data_frame = pd.read_csv('training.csv')

    data_frame['Image'] = data_frame['Image'].apply(lambda i: np.fromstring(i, sep=' '))
    data_frame = data_frame.dropna()  # Get only the data with 15 keypoints

    # Extract Images pixel values
    imgs_array = np.vstack(data_frame['Image'].values) / 255.0
    imgs_array = imgs_array.astype(np.float32)  # Normalize, target values to (0, 1)
    imgs_array = imgs_array.reshape(-1, 96, 96, 1)

    # Extract labels (key point cords)
    labels_array = data_frame[data_frame.columns[:-1]].values
    labels_array = (labels_array - 48) / 48  # Normalize, traget cordinates to (-1, 1)
    labels_array = labels_array.astype(np.float32)

    # shuffle the train data
    #     imgs_array, labels_array = shuffle(imgs_array, labels_array, random_state=9)

    return imgs_array, labels_array


# from keras.optimizers import Adam


# Main model
def the_model():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu',
                     input_shape=X_train.shape[1:]))  # Input shape: (96, 96, 1)
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    # Convert all values to 1D array
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(30))

    return model


X_train, y_train = data_loader()
print("Training datapoint shape: X_train.shape:{}".format(X_train.shape))
print("Training labels shape: y_train.shape:{}".format(y_train.shape))

epochs = 60
batch_size = 64

model = the_model()
hist = History()

checkpointer = ModelCheckpoint(filepath='checkpoint1.hdf5',
                               verbose=1, save_best_only=True)

# Complie Model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model_fit = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                      callbacks=[checkpointer, hist], verbose=1)

model.save('model1.h5')