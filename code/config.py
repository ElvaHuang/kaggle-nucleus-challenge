import os
import random
import numpy as np

''' Directories '''
root_dir = '/Users/rongyao.huang/Projects/kaggle-nucleus-challenge'
input_dir = os.path.join(root_dir, 'input')
intermediate_dir = os.path.join(root_dir, 'intermediate')
output_dir = os.path.join(root_dir, 'output')
model_dir = os.path.join(root_dir, 'models')
log_dir = os.path.join(root_dir, 'logs')

''' Preprocessor'''
IMG_CHANNELS = 3
IMG_WIDTH = 256
IMG_HEIGHT = 256

''' Evaluator '''
smooth = 1.0

''' Seed '''
seed = 42
random.seed = seed
np.random.seed = seed
