import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import re
import numpy as np
from util import resize_and_normalize, generate_img, preprocess_image_test

DATASET_DIR = sys.argv[1]
DATASET = re.match(r'(.*)_\d+', DATASET_DIR).group(1)

G = tf.saved_model.load(f'results/{DATASET_DIR}/G')
F = tf.saved_model.load(f'results/{DATASET_DIR}/F')

test_A = plt.imread(f'dino.jpg')
test_A = resize_and_normalize(test_A)
gen_A = generate_img(G, test_A)

test_B = plt.imread(f'tests/{DATASET}/B.jpg')
test_B = resize_and_normalize(test_B)
gen_B = generate_img(F, test_B)
plt.imsave(f'tests/gen_{DATASET}.jpg', np.concatenate((gen_A, gen_B)))

# cmd: /home/jglane/.conda/envs/cent7/2020.11-py38/my_tf_env/bin/python test.py apple2orange
# http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/

# Using the testing dataset
ds, ds_info = tfds.load(f'cycle_gan/{DATASET}', with_info=True, as_supervised=True)
ds_test_A = ds['testA']
ds_test_B = ds['testB']
ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)
