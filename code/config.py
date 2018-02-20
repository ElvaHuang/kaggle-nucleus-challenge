import os


''' Directories '''
root_dir = '/Users/rongyao.huang/Projects/kaggle-nucleus-challenge'
input_dir = os.path.join(root_dir, 'input')
intermediate_dir = os.path.join(root_dir, 'intermediate')
output_dir = os.path.join(root_dir, 'output')


''' Preprocessor'''
IMG_CHANNELS = 3

''' Evaluator '''
smooth = 1.0