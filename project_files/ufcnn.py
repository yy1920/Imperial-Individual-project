
import tensorflow as tf
from keras import Input
from keras.layers.convolutional import Convolution1D, ZeroPadding1D


def ufcnn_model(sequence_length=10,
                       features=1,
                       num_filter=50,
                       filter_length=2,
                       output_size=1,
                       optimizer='adam',
                       activation="relu",
                       init="glorot_normal"):

    inputs = Input(name='input', shape=(sequence_length,features))
    #x = ZeroPadding1D(2, name='input_padding') (inputs)# to avoid lookahead bias

    start = Convolution1D(filters=num_filter, kernel_size=filter_length, activation=activation, padding='valid', kernel_initializer=init, input_shape=(sequence_length, features),name='conv1')(inputs)
    start2 = Convolution1D(filters=num_filter, kernel_size=filter_length, activation =activation, padding='same', kernel_initializer=init, name='conv2')(start)
    x = Convolution1D(filters=num_filter, kernel_size=filter_length, activation=activation, padding='same', kernel_initializer=init, name='conv3')(start2)
    x = Convolution1D(filters=num_filter, kernel_size=filter_length, activation=activation, padding='same', kernel_initializer=init, name='conv4')(x)
    x = tf.keras.layers.concatenate([start2, x], axis=-1)
    x = Convolution1D(filters=num_filter, kernel_size=filter_length, activation=activation, padding='same', kernel_initializer=init, name='conv5')(x)
    x = tf.keras.layers.concatenate([start, x], axis=-1)
    x = Convolution1D(filters=num_filter, kernel_size=filter_length, activation=activation, padding='same', kernel_initializer=init, name='conv6')(x)
    x = Convolution1D(filters=num_filter, kernel_size=filter_length, padding='same', kernel_initializer=init, name='conv7')(x)
    x = Convolution1D(filters=output_size, kernel_size=filter_length, padding='same', kernel_initializer=init, name='conv8')(x)

    #x = Activation('relu', name='relu7')(x)

    # y = LSTM(128,return_sequences=True)(inputs)
    # y = LSTM(64)(y)
    # # y = Dense(1)(y)
    # x = tf.keras.layers.concatenate([x, y], axis=-1)
    #x = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss='mse')

    return model

