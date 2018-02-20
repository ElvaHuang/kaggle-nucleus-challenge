import numpy as np
from keras import backend as K

from config import smooth


def rle(mask_matrix):
    """
    Given a mask matrix of shape (a,b) filled with 1s and 0s,
    convert to a list of run length encoded labels for submission.
    :param mask_matrix:
    :return:
    """
    dots = np.where(mask_matrix.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b+1, 0))

        run_lengths[-1] += 1
        prev = b

    return run_lengths


def dice_coef(y_true, y_pred, smooth=smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)
