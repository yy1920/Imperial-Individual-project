import tensorflow as tf
from keras import Input, layers
from keras.layers import LSTM, AlphaDropout, Dense, Activation
from keras.layers.convolutional import Convolution1D, ZeroPadding1D
from tensorflow.python.keras.layers import Flatten, Dropout
tf.random.set_seed(1234)
def sum_mse_loss(y_true, y_pred):
    sum =tf.reshape(tf.math.reduce_sum(y_pred),shape=(tf.shape(y_true)))
    # print(tf.math.reduce_sum(y_pred))
    # print(sum)
    #return tf.square(y_true-sum)

    return tf.keras.losses.MeanSquaredError()(y_true,sum)

class weighted_layer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # I've added pass because this is the simplest form I can come up with.
        pass

    def call(self, inputs):
        inp1, inp2 = inputs
        inp2 = tf.reshape(inp2,shape=(-1,1))
        return tf.math.multiply(inp1,inp2)


def cnnlstm_model(sequence_length=10,
                  features=1,
                  filter_length=3,
                  output_size=1,
                  filter_num = 25,
                  optimizer='adam',
                  activation=tf.keras.activations.selu,
                  init="glorot_normal"):
    inputs = Input(name='input', shape=(sequence_length, features))
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
    dense1 = Dense(10)(x)
    w1 = weighted_layer()([inputs, dense1])
    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=sum_mse_loss, metrics=["mae", "mape"])
    model.summary()
    return model
