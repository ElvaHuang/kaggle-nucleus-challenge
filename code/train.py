import sys
sys.path.append('/Users/rongyao.huang/Projects/kaggle-nucleus-challenge/code')

from datetime import datetime

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from preprocessors import load_train_val_data
from augmentor import datagen
from models import unet_1, unet_2

from config import *

run_id = str(datetime.utcnow())
tb = TensorBoard(log_dir=os.path.join(log_dir, run_id), histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=4, min_lr=1e-6)


def train(resume=False):

    print('Loading data...')
    X_train, X_val, y_train, y_val = load_train_val_data()
    print('X_train of shape: {}\nX_val of shape: {}'.format(X_train.shape, X_val.shape))
    print('y_train of shape: {}\ny_val of shape: {}'.format(y_train.shape, y_val.shape))

    print('Creating and compiling model...')
    model = unet_1()

    if resume:
        model.load_weights(os.path.join(model_dir, 'unet_2.hdf5'))

    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'unet_2_{}.hdf5'.format(run_id)),
                                       monitor='val_loss', save_best_only=True)
    earlystopper = EarlyStopping(patience=5, verbose=1)
    model.summary()

    print('Creating data generator as model input...')
    batch_size = 16
    nb_epoch = 200

    datagen.fit(X_train)
    train_generator = datagen.flow(X_train, np.expand_dims(y_train, axis=-1), batch_size=batch_size)
    val_generator = datagen.flow(X_val, np.expand_dims(y_val, axis=-1), batch_size=batch_size)

    # Use fixed samples instead to visualize histograms. There is currently a bug that prevents it
    # when a val generator is used.
    # Not aug val samples to keep the eval consistent.

    print('Training model...')
    model.fit_generator(train_generator, steps_per_epoch=X_train.shape[0], epochs=nb_epoch,
                        validation_data=val_generator, validation_steps=X_val.shape[0],
                        verbose=2, callbacks=[model_checkpoint, earlystopper, reduce_lr, tb])

if __name__ == '__main__':
    train(resume=False)
