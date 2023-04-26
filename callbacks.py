import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from util import generate_img, preprocess_image_test

class GenRandomImg(tf.keras.callbacks.Callback):
    def __init__(self, ds_test_A, ds_test_B, results_dir):
        super(GenRandomImg, self).__init__()
        self.ds_test_A = ds_test_A
        self.ds_test_B = ds_test_B
        self.results_dir = results_dir

    def on_epoch_end(self, epoch, logs=None):
        test_A = next(iter(self.ds_test_A))[0]
        test_B = next(iter(self.ds_test_B))[0]
        gen_A = generate_img(self.model.G, test_A)
        gen_B = generate_img(self.model.F, test_B)
        plt.imsave(f'{self.results_dir}/img/epoch_{epoch + 1}.png', np.concatenate((gen_A, gen_B)))

class GenSameImg(tf.keras.callbacks.Callback):
    def __init__(self, image_set_dir, results_dir):
        super(GenSameImg, self).__init__()
        self.image_set_dir = image_set_dir
        self.results_dir = results_dir

    def on_epoch_end(self, epoch, logs=None):
        test_imgs = sorted(os.listdir(self.image_set_dir))
        test_A = plt.imread(f'{self.image_set_dir}/{test_imgs[0]}')
        test_B = plt.imread(f'{self.image_set_dir}/{test_imgs[1]}')
        test_A = preprocess_image_test(test_A, None)
        test_B = preprocess_image_test(test_B, None)
        gen_A = generate_img(self.model.G, test_A)
        gen_B = generate_img(self.model.F, test_B)
        plt.imsave(f'{self.results_dir}/img/epoch_{epoch + 1}.png', np.concatenate((gen_A, gen_B)))