import tensorflow as tf
from keras import Input
from keras.layers import LSTM, AlphaDropout, Dense, Activation
from keras.layers.convolutional import Convolution1D, ZeroPadding1D
from tensorflow.python.keras.layers import Flatten, Dropout


def cnnlstm_model(sequence_length=10,
                  features=1,
                  filter_length=3,
                  output_size=1,
                  filter_num = 25,
                  optimizer='adam',
                  activation=tf.keras.activations.selu,
                  init="glorot_normal"):
    inputs = Input(name='input', shape=(sequence_length, features))

    # x = Convolution1D(filters=filter_num, kernel_size=filter_length, activation=activation, padding='valid',
    #                   kernel_initializer=init, input_shape=(sequence_length, features), name='conv1')(inputs)
    # x = AlphaDropout(0.2)(x)
    # # x = Convolution1D(filters=filter_num, kernel_size=filter_length, activation=activation, padding='valid',
    # #                   kernel_initializer=init, input_shape=(sequence_length, features), name='conv2')(x)
    #
    # # x = AlphaDropout(0.2)(x)
    # x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    # x = Convolution1D(filters=filter_num, kernel_size=int(filter_length/2), activation=activation, padding='valid',
    #                   kernel_initializer=init, input_shape=(sequence_length, features), name='conv3')(x)
    # x = AlphaDropout(0.2)(x)
    # x = Convolution1D(filters=filter_num, kernel_size=int(filter_length/2), activation=activation, padding='valid',
    #                   kernel_initializer=init, input_shape=(sequence_length, features), name='conv4')(x)
    # x = AlphaDropout(0.2)(x)
    # x = Convolution1D(filters=filter_num, kernel_size=int(filter_length/2), padding='valid',
    #                   kernel_initializer=init, input_shape=(sequence_length, features), name='convF')(x)
    # # x = LSTM(64, activation=activation, return_sequences=True)(x)
    # # x = AlphaDropout(0.5)(x)
    # x = LSTM(64, activation=activation)(x)
    # x = Dense(1)(x)
    x = Convolution1D(filters=64, kernel_size=filter_length, activation=activation, padding='valid',
                       kernel_initializer=init, input_shape=(sequence_length, features), name='conv1')(inputs)
    x = AlphaDropout(0.05)(x)
    x = Convolution1D(filters=32, kernel_size=filter_length, activation=activation, padding='valid',
                      kernel_initializer=init, input_shape=(sequence_length, features), name='conv2')(inputs)
    x = AlphaDropout(0.05)(x)

    x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
    x = Convolution1D(filters=16, kernel_size=filter_length, activation=activation, padding='valid',
                      kernel_initializer=init, input_shape=(sequence_length, features), name='conv3')(inputs)
    x = AlphaDropout(0.05)(x)
    x = Convolution1D(filters=16, kernel_size=filter_length, activation=activation, padding='valid',
                      kernel_initializer=init, input_shape=(sequence_length, features), name='conv3')(inputs)
    x = LSTM(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    #x = Flatten()(x)
    x = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model
