# reference: https://github.com/raghakot/ultrasound-nerve-segmentation/blob/master/submission.py

import os
import numpy as np

from datetime import datetime
from itertools import chain

from preprocessors import load_test_data
from skimage.transform import resize
from models import unet_1, unet_2

# todo: not runnable yet

def post_process_mask(prob_mask, orig_h, orig_w):
    """
    Smoothens the mask probs, upsamples to original size, and thresholds.
    """
    prob_mask = prob_mask.astype('float32')
    prob_mask = resize(prob_mask, (orig_h, orig_w))

    # # To get smooth mask shape
    # prob_mask = cv2.GaussianBlur(prob_mask, (51, 51), 0)
    #
    # prob_mask = cv2.threshold(prob_mask, 0.5, 1, cv2.THRESH_BINARY)[1]
    return prob_mask


def run_length_enc(label):
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def generate_submission():
    # Load test images and preprocess for conv net.
    print('Loading and processing test images')
    imgs_test, orig_sz_test = load_test_data()
    total = imgs_test.shape[0]
    # imgs = np.ndarray((total, 1, DataManager.IMG_TARGET_ROWS, DataManager.IMG_TARGET_ROWS), dtype=np.uint8)
    # i = 0
    # for img in imgs_test:
    #     imgs[i] = preprocess(img)
    #     i += 1

    print('Loading model...')
    model = unet_1()
    model.load_weights(os.path.join(model, 'unet_1.hdf5'))

    print('Generating predictions...')
    masks = model.predict(imgs_test, verbose=1)

    ids = []
    rles = []
    for i in range(total):

        mask = post_process_mask(masks[i])
        rle = run_length_enc(mask)
        rles.append(rle)
        ids.append(i + 1)

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'img,pixels'
    file_name = 'results/submission_{}.csv'.format(str(datetime.now()))

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    generate_submission()