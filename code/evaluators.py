import numpy as np
from keras import backend as K
import tensorflow as tf

from config import smooth

# reference: https://www.kaggle.com/c/data-science-bowl-2018/discussion/51553


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


def iou_coef(y_true, y_pred, smooth=smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def iou_loss(y_true, y_pred, smooth=smooth):
    return -iou_coef(y_true, y_pred)


def mean_iou(y_true, y_pred, smooth=smooth):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    ious = (intersection + smooth) / (union + smooth)

    metrics = []
    for t in np. arange(0.5, 0.95, 0.05):
        ious_ = tf.to_int32(ious > t)
        metrics.append(K.sum(ious_)/tf.shape(ious_)[0])

    return K.mean(metrics)
