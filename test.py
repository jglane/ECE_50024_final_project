import tensorflow as tf
import matplotlib.pyplot as plt
from util import *
import sys

args = sys.argv

image_set = args[1]
G = tf.saved_model.load(f'results/{image_set}/G')
F = tf.saved_model.load(f'results/{image_set}/F')

test_img = plt.imread('test/o.jpg')
test_img = resize_and_normalize(test_img)
gen_img = generate_img(F, test_img)
plt.imsave('testo.png', gen_img)

# cmd: /home/jglane/.conda/envs/cent7/2020.11-py38/my_tf_env/bin/python test.py apple2orange