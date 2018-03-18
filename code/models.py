import sys
sys.path.append('/Users/rongyao.huang/Projects/kaggle-nucleus-challenge/code')

from keras.layers import (
    Input,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    UpSampling2D,
    MaxPooling2D,
    Conv2DTranspose
)
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import ELU
from keras import Sequential
from keras.models import Model
from keras.optimizers import Adam

from evaluators import iou_loss, iou_coef

from config import *


def unet_1(optimizer=Adam(lr=1e-4)):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


def unet_2(optimizer=Adam(lr=1e-4)):

    # reference: https://github.com/raghakot/ultrasound-nerve-segmentation

    # helper function to build conv -> batchnorm -> elu blocks
    def _conv_bn_relu(nb_filter, nb_row, nb_col, strides=(1, 1)):
        def f(input):
            conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col),
                                 strides=strides, kernel_initializer="he_normal",
                                 padding='same')(input)
            norm = BatchNormalization()(conv)
            return ELU()(norm)

        return f

    # contraction path
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), name='main_input')
    conv1 = _conv_bn_relu(32, 7, 7)(inputs)
    conv1 = _conv_bn_relu(32, 3, 3)(conv1)
    pool1 = _conv_bn_relu(32, 2, 2, strides=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = _conv_bn_relu(64, 3, 3)(drop1)
    conv2 = _conv_bn_relu(64, 3, 3)(conv2)
    pool2 = _conv_bn_relu(64, 2, 2, strides=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = _conv_bn_relu(128, 3, 3)(drop2)
    conv3 = _conv_bn_relu(128, 3, 3)(conv3)
    pool3 = _conv_bn_relu(128, 2, 2, strides=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = _conv_bn_relu(256, 3, 3)(drop3)
    conv4 = _conv_bn_relu(256, 3, 3)(conv4)
    pool4 = _conv_bn_relu(256, 2, 2, strides=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = _conv_bn_relu(512, 3, 3)(drop4)
    conv5 = _conv_bn_relu(512, 3, 3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    # expansion path
    up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop5), conv4], axis=3)
    conv6 = _conv_bn_relu(256, 3, 3)(up6)
    conv6 = _conv_bn_relu(256, 3, 3)(conv6)
    drop6 = Dropout(0.5)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop6), conv3], axis=3)
    conv7 = _conv_bn_relu(128, 3, 3)(up7)
    conv7 = _conv_bn_relu(128, 3, 3)(conv7)
    drop7 = Dropout(0.5)(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop7), conv2], axis=3)
    conv8 = _conv_bn_relu(64, 3, 3)(up8)
    conv8 = _conv_bn_relu(64, 3, 3)(conv8)
    drop8 = Dropout(0.5)(conv8)

    up9 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(drop8), conv1], axis=3)
    conv9 = _conv_bn_relu(32, 3, 3)(up9)
    conv9 = _conv_bn_relu(32, 3, 3)(conv9)
    drop9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name='main_output')(drop9)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # # Using conv to mimic fully connected layer.
    # aux = Conv2D(1, (drop5._keras_shape[2], drop5._keras_shape[3]),
    #                     strides=(1, 1), kernel_initializer="he_normal", activation='sigmoid')(drop5)
    # aux = Flatten(name='aux_output')(aux)

    # model = Model(input=inputs, output=[conv10, aux])
    # model.compile(optimizer=optimizer,
    #               loss={'main_output': iou_loss, 'aux_output': 'binary_crossentropy'},
    #               metrics={'main_output': iou_coef, 'aux_output': 'acc'},
    #               loss_weights={'main_output': 1, 'aux_output': 0.5})

    return model


def toy_cnn(optimizer=Adam(lr=1e-4)):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(None, None, IMG_CHANNELS),
                                      name='NormalizeInput'))
    model.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
    model.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
    # use dilations to get a slightly larger field of view
    model.add(Conv2D(16, kernel_size=(3, 3), dilation_rate=2, padding='same'))
    model.add(Conv2D(16, kernel_size=(3, 3), dilation_rate=2, padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), dilation_rate=3, padding='same'))

    # the final processing
    model.add(Conv2D(16, kernel_size=(1, 1), padding='same'))
    model.add(Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid'))
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics='acc')

    return model


