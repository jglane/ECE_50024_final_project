import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import numpy as np
from util import resize_and_normalize, generate_img, preprocess_image_test

args = sys.argv

if args[1] == 'summer2winter_yosemite':
    image_set = 'summer2winter'
else:
    image_set = args[1]

set_list = image_set.split('2')

G = tf.saved_model.load(f'results/{image_set}/G')
F = tf.saved_model.load(f'results/{image_set}/F')

test_A = plt.imread(f'tests/test_{set_list[0]}.jpg')
test_A = resize_and_normalize(test_A)
gen_A = generate_img(G, test_A)

test_B = plt.imread(f'tests/test_{set_list[1]}.jpg')
test_B = resize_and_normalize(test_B)
gen_B = generate_img(F, test_B)
plt.imsave(f'test/gen_{image_set}.jpg', np.concatenate((gen_A, gen_B)))

# cmd: /home/jglane/.conda/envs/cent7/2020.11-py38/my_tf_env/bin/python test.py apple2orange
# http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/

# Using the testing dataset
ds, ds_info = tfds.load(f'cycle_gan/{image_set}', with_info=True, as_supervised=True)
ds_test_A = ds['testA']
ds_test_B = ds['testB']
ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)
