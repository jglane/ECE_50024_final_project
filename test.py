import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np
from util import resize_and_normalize, generate_img

args = sys.argv

image_set = args[1]
set_list = image_set.split('2')

G = tf.saved_model.load(f'results/{image_set}/G')
F = tf.saved_model.load(f'results/{image_set}/F')

test_A = plt.imread(f'test/test_{set_list[0]}.jpg')
test_A = resize_and_normalize(test_A)
gen_A = generate_img(G, test_A)

test_B = plt.imread(f'test/test_{set_list[1]}.jpg')
test_B = resize_and_normalize(test_B)
gen_B = generate_img(F, test_B)
plt.imsave(f'test/gen_{image_set}.jpg', np.concatenate((gen_A, gen_B)))

# cmd: /home/jglane/.conda/envs/cent7/2020.11-py38/my_tf_env/bin/python test.py apple2orange