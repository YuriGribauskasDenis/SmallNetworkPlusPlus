import tensorflow as tf
import keras
import keras.backend as BCKN
from keras.models import Model, Sequential
from keras.layers import Input, Concatenate
from keras.layers import Dense, Conv2D, DepthwiseConv2D, Dropout, Flatten
from keras.layers import GlobalAveragePooling2D, Reshape
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import ReLU, Softmax, LeakyReLU
from keras.layers import BatchNormalization as BN
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils.vis_utils import plot_model
from keras import regularizers


# def preprocess_image(image, size):
#     image = keras.preprocessing.image.smart_resize(
#         image, size, interpolation='bilinear'
#     )
#     return image


def small_network():

    first = Input((100,100,1))
    model = MaxPooling2D(pool_size=2) (first)

    lambd = 0.1
    reglar = regularizers.L1L2(l1=1e-5, l2=1e-4)

    model = Conv2D (16, kernel_size=3, strides=1, padding='same', kernel_regularizer=reglar) (model)
    model = ReLU() (model)
    model = Conv2D (16, kernel_size=3, strides=1, padding='same', kernel_regularizer=reglar) (model)
    model = ReLU() (model)
    model = AveragePooling2D(pool_size=2) (model)

    model = Conv2D (24, kernel_size=3, strides=1, padding='same', kernel_regularizer=reglar) (model)
    model = ReLU() (model)
    model = Conv2D (24, kernel_size=3, strides=1, padding='same', kernel_regularizer=reglar) (model)
    model = ReLU() (model)
    model = AveragePooling2D(pool_size=(2, 2)) (model)

    model = Conv2D (32, kernel_size=3, strides=1, padding='same', kernel_regularizer=reglar) (model)
    model = GlobalAveragePooling2D() (model)
    model = Flatten() (model)
    model = Dense(3) (model)
    last = Softmax() (model)

    return Model(first, last)


BCKN.clear_session()
model = small_network()
model.summary()


plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
# optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.0)
optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True, clipnorm=1.0)
# loss_fn = 'categorical_crossentropy'
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])


import numpy as np
X  = np.random.rand(160, 100, 100, 1)
y = np.random.randint(2, size=160)


reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,
                              patience=5, min_lr=0.001)
early_stop = EarlyStopping(monitor='loss', patience=3)


history = model.fit(X, y, epochs=15, batch_size=16, callbacks=[early_stop, reduce_lr], verbose=1)
len(history.history['loss'])