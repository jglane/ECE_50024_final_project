import tensorflow as tf
import matplotlib.pyplot as plt
from util import *

image_set = 'apple2orange'
G = tf.saved_model.load(f'results/{image_set}/G')
F = tf.saved_model.load(f'results/{image_set}/F')

test_img = plt.imread('test/o.jpg')
test_img = resize_and_normalize(test_img)
gen_img = generate_img(F, test_img)
plt.imsave('testo.png', gen_img)