import numpy as np
import pandas as pd
from skimage.io import imread
from glob import glob
import os

from utils import timer

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

    return img_meta_df


@timer
def extract_from_meta(img_meta_df, ch, type='train'):

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

    def _read_and_stack(list_imgs):
        """
        given a list of paths to images:
        - read all in as numpy arrays
        - stack along 1st dimension
        - in the case of multiple masks of the same image, sum along 1st dimension to combine them
        - apply normalization by divide each cell with 255.
        """
        return np.sum(np.stack([imread(i) for i in list_imgs], 0), 0) / 255.0

    tgt_df['images'] = tgt_df['images'].map(_read_and_stack).map(lambda x: x[:, :, :ch])
    try:
        tgt_df['masks'] = tgt_df['masks'].map(_read_and_stack).map(lambda x: x.astype(int))
    except ValueError:
        tgt_df['masks'] = None

    return tgt_df


@timer
def get_train_rle_labels(path_input, stage='stage1'):

    print('Extracting Run Length Encoded labels for training ...')

    train_rle_labels = pd.read_csv(os.path.join(path_input, '{}_train_labels.csv'.format(stage)))
    train_rle_labels['EncodedPixels'] = train_rle_labels['EncodedPixels']\
        .map(lambda p: [int(x) for x in p.split(' ')])
    return train_rle_labels


if __name__ == '__main__':
    from config import input_dir, intermediate_dir, IMG_CHANNELS

    img_meta_df = get_img_metadata(input_dir, ch=IMG_CHANNELS, IMstage='stage1')
    img_meta_df.to_json(os.path.join(intermediate_dir, 'img_meta_df.json'))
    # img_meta_df.to_pickle(os.path.join(intermediate_dir, 'img_meta_df.pkl'))

    tgt_train_df = extract_from_meta(img_meta_df, ch=IMG_CHANNELS, type='train')
    tgt_train_df.to_json(os.path.join(intermediate_dir, 'tgt_train_df.json'))
    # tgt_train_df.to_pickle(os.path.join(intermediate_dir, 'tgt_train_df.pkl'))    # too big to pickle?

    tgt_test_df = extract_from_meta(img_meta_df, ch=IMG_CHANNELS, type='test')
    tgt_test_df.to_json(os.path.join(intermediate_dir, 'tgt_test_df.json'))
    # tgt_test_df.to_pickle(os.path.join(intermediate_dir, 'tgt_test_df.pkl'))      # too big to pickle?

    train_rle_labels_df = get_train_rle_labels(input_dir, stage='stage1')
    train_rle_labels_df.to_json(os.path.join(intermediate_dir, 'train_rle_labels_df.json'))
    # train_rle_labels_df.to_pickle(os.path.join(intermediate_dir, 'train_rle_labels_df.pkl'))






