from keras import Input, Model
from keras.layers import LSTM, Dense, BatchNormalization, MaxPool1D
from keras.layers.convolutional import Convolution1D, ZeroPadding1D
from tensorflow.python.keras.layers import Flatten, Dropout
from tensorflow import random

random.set_seed(1234)


def cnnlstm_model(sequence_length=10,
                  features=1,
                  filter_length=3,
                  output_size=1,
                  filter_num=25,
                  optimizer='adam',
                  activation='relu',
                  init="glorot_normal"):
    inputs = Input(name='input', shape=(sequence_length, features))
    x = Convolution1D(filters=16, kernel_size=filter_length, activation=activation, padding='valid',
                      kernel_initializer=init, input_shape=(sequence_length, 1), name='conv4')(inputs)
    x = BatchNormalization()(x)
    x = LSTM(64, activation='relu', name='lstm1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    return model
