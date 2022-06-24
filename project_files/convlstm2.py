import tensorflow as tf
from keras import Input
from keras.layers import LSTM, Dense
from keras.layers.convolutional import Convolution1D
from tensorflow.python.keras.layers import Dropout

tf.random.set_seed(1234)


def cnnlstm_model(sequence_length=10,
                  features=1,
                  filter_length=3,
                  output_size=1,
                  filter_num=25,
                  optimizer='adam',
                  activation=tf.keras.activations.selu,
                  init="glorot_normal"):
    inputs = Input(name='input', shape=(sequence_length, features))
    x = Convolution1D(filters=16, kernel_size=filter_length, activation=activation, padding='valid',
                      kernel_initializer=init, input_shape=(sequence_length, features), name='conv3')(inputs)
    x = LSTM(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss='mse', metrics=["mae", "mape"])
    model.summary()
    return model

## INPUT --> CNN --> LSTM --> DROPOUT --> DENSE --> OUTPUT
