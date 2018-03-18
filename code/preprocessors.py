import sys
sys.path.append('/Users/rongyao.huang/Projects/kaggle-nucleus-challenge/code')

import os
from glob import glob
import numpy as np

import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split

from utils import timer
from config import *

@timer
def get_img_metadata(path_input, stage='stage1'):
    """ Gather metadata of all available train/test images
        into a dataframe for easy access
    """

    print('Gathering image metadata from path: {} ...'.format(path_input))
    all_imgs = glob(os.path.join(path_input, '{}_*'.format(stage), '*', '*', '*'))
    img_meta_df = pd.DataFrame({'path': all_imgs})

    get_id = lambda path: path.split('/')[-3]
    get_type = lambda path: path.split('/')[-2]
    get_grp = lambda path: path.split('/')[-4].split('_')[1]
    get_stg = lambda path: path.split('/')[-4].split('_')[0]

    img_meta_df['image_id'] = img_meta_df['path'].map(get_id)
    img_meta_df['image_type'] = img_meta_df['path'].map(get_type)
    img_meta_df['train_test_split'] = img_meta_df['path'].map(get_grp)
    img_meta_df['stage'] = img_meta_df['path'].map(get_stg)

    print('done.')

    return img_meta_df


@timer
def extract_from_meta(img_meta_df, new_img_ch, if_resize=True, type='train', **kwargs):

    print('Extracting data for {}ing ...'.format(type))

    tgt_meta_df = img_meta_df.query('train_test_split=="{}"'.format(type))

    groupby_cols = ['stage', 'image_id']
    all_rows = []
    for grp, records in tgt_meta_df.groupby(groupby_cols):
        row = {grp_name: grp_val for grp_name, grp_val in zip(groupby_cols, grp)}
        row['masks'] = records.query('image_type=="masks"')['path'].values.tolist()
        row['images'] = records.query('image_type=="images"')['path'].values.tolist()
        all_rows += [row]
    tgt_df = pd.DataFrame(all_rows)

    if if_resize:
        new_img_h = kwargs.get('new_img_h', 128)
        new_img_w = kwargs.get('new_img_w', 128)
        print('Resize images to ({}, {})'.format(new_img_h, new_img_w))

        def _resize(img):
            return resize(img, (new_img_h, new_img_w), mode='constant', preserve_range=True)

    def _read_and_stack(list_imgs, if_resize=if_resize):
        """
        given a list of paths to images:
        - read all in as numpy arrays
        - stack along 1st dimension
        - in the case of multiple masks of the same image, sum along 1st dimension to combine them
        - apply normalization by dividing each cell with 255.
        """
        if if_resize:
            out = np.sum(np.stack([_resize(imread(i)) for i in list_imgs], 0), 0) / 255.0
        else:
            out = np.sum(np.stack([imread(i) for i in list_imgs], 0), 0) / 255.0
        return out

    tgt_df['images'] = tgt_df['images'].map(_read_and_stack).map(lambda x: x[:, :, :new_img_ch])
    tgt_df['orig_sizes'] = tgt_df['images'].map(lambda x: (x.shape[0], x.shape[1]))

    try:
        tgt_df['masks'] = tgt_df['masks'].map(_read_and_stack).map(lambda x: x.astype(int))

    except ValueError:
        tgt_df['masks'] = None

    print('done.')

    return tgt_df

@timer
def df2array(df, type='train'):
    if type == 'train':
        X_train = np.stack(df['images'].as_matrix(), axis=0)
        y_train = np.stack(df['masks'].as_matrix(), axis=0)
        # orig_sizes_train = df['orig_sizes'].as_matrix()
        return X_train, y_train

    if type == 'test':
        X_test = np.stack(df['images'].as_matrix(), axis=0)
        orig_sizes_test = np.stack(df['orig_sizes'].as_matrix(), axis=0)
        return X_test, orig_sizes_test


@timer
def train_val_split(X, y, seed=1234):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, X_val, y_train, y_val


@timer
def save_data(X_train, y_train, X_val, y_val, X_test, orig_sz_test):
    np.save(os.path.join(intermediate_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(intermediate_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(intermediate_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(intermediate_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(intermediate_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(intermediate_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(intermediate_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(intermediate_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(intermediate_dir, 'orig_sz_test.npy'), orig_sz_test)
    print('Saving .npy files done.')


@timer
def load_train_val_data():
    X_train = np.load(os.path.join(intermediate_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(intermediate_dir, 'X_val.npy'))
    y_train = np.load(os.path.join(intermediate_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(intermediate_dir, 'y_val.npy'))

    print('Loading training and validation data done.')

    return X_train, X_val, y_train, y_val


@timer
def load_test_data():
    X_test = np.load(os.path.join(intermediate_dir, 'X_test.npy'))
    orig_sz_test = np.load(os.path.join(intermediate_dir, 'orig_sz_test.npy'))
    print('Loading test data done.')

    return X_test, orig_sz_test


@timer
def get_train_rle_labels(path_input, stage='stage1'):

    print('Extracting Run Length Encoded labels for training ...')

    train_rle_labels = pd.read_csv(os.path.join(path_input, '{}_train_labels.csv'.format(stage)))
    train_rle_labels['EncodedPixels'] = train_rle_labels['EncodedPixels']\
        .map(lambda p: [int(x) for x in p.split(' ')])
    return train_rle_labels


if __name__ == '__main__':

    img_meta_df = get_img_metadata(input_dir, stage='stage1')
    img_meta_df.to_json(os.path.join(intermediate_dir, 'img_meta_df.json'))

    tgt_train_df = extract_from_meta(img_meta_df, 3, if_resize=True, type='train', new_img_h=256, new_img_w=256)
    tgt_test_df = extract_from_meta(img_meta_df, 3, if_resize=True, type='test', new_img_h=256, new_img_w=256)
    X_train_orig, y_train_orig = df2array(tgt_train_df, type='train')
    X_test, orig_sz_test = df2array(tgt_train_df, type='test')

    X_train, X_val, y_train, y_val = train_val_split(X_train_orig, y_train_orig)
    save_data(X_train, y_train, X_val, y_val, X_test, orig_sz_test)

    # train_rle_labels_df = get_train_rle_labels(input_dir, stage='stage1')
    # train_rle_labels_df.to_json(os.path.join(intermediate_dir, 'train_rle_labels_df.json'))
    # # train_rle_labels_df.to_pickle(os.path.join(intermediate_dir, 'train_rle_labels_df.pkl'))
