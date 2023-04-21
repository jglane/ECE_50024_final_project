import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from util import generate_img

class GenImg(tf.keras.callbacks.Callback):
    def __init__(self, ds_test_A, ds_test_B, image_set):
        super(GenImg, self).__init__()
        self.ds_test_A = ds_test_A
        self.ds_test_B = ds_test_B

        i = 0
        while f'{image_set}_{i}' in os.listdir('results'):
            i += 1
        self.dir = f'results/{image_set}_{i}'
        os.mkdir(self.dir)
        os.mkdir(f'{self.dir}/img')


    def on_epoch_end(self, epoch, logs=None):
        test_A = next(iter(self.ds_test_A))[0]
        test_B = next(iter(self.ds_test_B))[0]
        gen_A = generate_img(self.model.G, test_A)
        gen_B = generate_img(self.model.F, test_B)
        plt.imsave(f'{self.dir}/img/epoch_{epoch + 1}.png', np.concatenate((gen_A, gen_B)))