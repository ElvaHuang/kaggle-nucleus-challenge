from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, UpSampling2D, Lambda


def toy_cnn(ch):
    toy_cnn = Sequential()
    toy_cnn.add(BatchNormalization(input_shape=(None, None, ch),
                                      name='NormalizeInput'))
    toy_cnn.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
    toy_cnn.add(Conv2D(8, kernel_size=(3, 3), padding='same'))
    # use dilations to get a slightly larger field of view
    toy_cnn.add(Conv2D(16, kernel_size=(3, 3), dilation_rate=2, padding='same'))
    toy_cnn.add(Conv2D(16, kernel_size=(3, 3), dilation_rate=2, padding='same'))
    toy_cnn.add(Conv2D(32, kernel_size=(3, 3), dilation_rate=3, padding='same'))

    # the final processing
    toy_cnn.add(Conv2D(16, kernel_size=(1, 1), padding='same'))
    toy_cnn.add(Conv2D(1, kernel_size=(1, 1), padding='same', activation='sigmoid'))
    toy_cnn.summary()

    return toy_cnn