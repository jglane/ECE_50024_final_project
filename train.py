import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
from util import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMG_DIM = 256

image_set = 'maps'
ds, ds_info = tfds.load(f'cycle_gan/{image_set}', with_info=True, as_supervised=True)

ds_train_A = ds['trainA']
ds_train_B = ds['trainB']

ds_test_A = ds['testA']
ds_test_B = ds['testB']

ds_train_A = ds_train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainA'].num_examples).batch(1)
ds_train_B = ds_train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainB'].num_examples).batch(1)
ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)

class Residual(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides, padding='same', activation='relu'):
        super(Residual, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        out = tf.keras.layers.add([x, inputs])
        return tf.keras.layers.ReLU()(out)
    
def build_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((IMG_DIM, IMG_DIM, 3)))

    # c7s1-64
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=7, strides=1, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # d128
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # d256
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())
    
    # 9 x R256
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))
    model.add(Residual(filters=256, kernel_size=3, strides=1))

    # u128
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # u64
    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same'))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.ReLU())

    # c7s1-3
    model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same', activation='tanh'))

    return model

def build_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input((IMG_DIM, IMG_DIM, 3)))
    
    # C64
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2))
    model.add(tf.keras.layers.LeakyReLU(0.2))

    # C128
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    # C256
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))
    
    # C512
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=4, strides=2))
    model.add(tfa.layers.InstanceNormalization())
    model.add(tf.keras.layers.LeakyReLU(0.2))

    # Convolution to produce a 1D output
    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=4, activation='sigmoid'))

    return model

G = build_generator()
F = build_generator()

D_X = build_discriminator()
D_Y = build_discriminator()

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
lambd = 10

# Loss functions
def discriminator_loss(real, fake):
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) / 2

def generator_loss(fake):
    return loss(tf.ones_like(fake), fake)

def cycle_loss(real, cycled):
    return lambd * tf.reduce_mean(tf.abs(real - cycled))

def identity_loss(real, same):
    return lambd * tf.reduce_mean(tf.abs(real - same)) / 2

# Initialize optimizers
learning_rate = 0.0002
G_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
F_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
D_X_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
D_Y_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

# Custom training step
@tf.function()
def train_step(real_X, real_Y):
    with tf.GradientTape(persistent=True) as tape:
        # Generate images
        fake_Y = G(real_X, training=True)
        cycled_X = F(fake_Y, training=True)
        fake_X = F(real_Y, training=True)
        cycled_Y = G(fake_X, training=True)
        same_X = F(real_X, training=True)
        same_Y = G(real_Y, training=True)

        # Discriminator output
        D_X_real = D_X(real_X, training=True)
        D_X_fake = D_X(fake_X, training=True)
        D_Y_real = D_Y(real_Y, training=True)
        D_Y_fake = D_Y(fake_Y, training=True)

        # Calculate the loss
        total_cycle_loss = cycle_loss(real_X, cycled_X) + cycle_loss(real_Y, cycled_Y)
        G_loss = generator_loss(D_Y_fake) + identity_loss(real_Y, same_Y) + total_cycle_loss
        F_loss = generator_loss(D_X_fake) + identity_loss(real_X, same_X) + total_cycle_loss

        D_X_loss = discriminator_loss(D_X_real, D_X_fake)
        D_Y_loss = discriminator_loss(D_Y_real, D_Y_fake)

    # Calculate the gradients for generator and discriminator
    G_grad = tape.gradient(G_loss, G.trainable_variables)
    F_grad = tape.gradient(F_loss, F.trainable_variables)

    D_X_grad = tape.gradient(D_X_loss, D_X.trainable_variables)
    D_Y_grad = tape.gradient(D_Y_loss, D_Y.trainable_variables)

    # Apply the gradients to the optimizer
    G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
    F_opt.apply_gradients(zip(F_grad, F.trainable_variables))

    D_X_opt.apply_gradients(zip(D_X_grad, D_X.trainable_variables))
    D_Y_opt.apply_gradients(zip(D_Y_grad, D_Y.trainable_variables))

    return G_loss, F_loss, D_X_loss, D_Y_loss

# Visualize the training process
i = 0
while f'{image_set}_{i}' in os.listdir('results'):
    i += 1
dir = f'results/{image_set}_{i}'
os.mkdir(dir)
os.mkdir(f'{dir}/img')

# Redirect stdout to file
sys.stdout = open(f'{dir}/log.txt', 'w')

# Train the model
epochs = 100
losses_list = []
for epoch in range(epochs):
    start = time.time()

    # Loop through the training set
    print(f'Epoch: {epoch + 1}/{epochs} ', end='')
    for X, Y in tf.data.Dataset.zip((ds_train_A, ds_train_B)):
        G_loss, F_loss, D_X_loss, D_Y_loss = train_step(X, Y)
        print('.', end='')
    
    # Choose random images from the test set
    test_A = next(iter(ds_test_A))[0]
    test_B = next(iter(ds_test_B))[0]
    gen_A = generate_img(G, test_A)
    gen_B = generate_img(F, test_B)
    plt.imsave(f'{dir}/img/epoch_{epoch + 1}.png', np.concatenate((gen_A, gen_B)))

    # Log the losses for each epoch
    end = time.time()
    elapsed = np.round(end - start, 1)
    print(f'\n    G_loss: {G_loss:.4f}    F_loss: {F_loss:.4f}    D_X_loss: {D_X_loss:.4f}    D_Y_loss: {D_Y_loss:.4f}    Time: {elapsed} s\n')

    # Save the losses for plotting
    losses_list.append(np.array([G_loss, F_loss, D_X_loss, D_Y_loss]))

# Plot the losses
losses_array = np.array(losses_list)
plt.figure()
plt.plot(losses_array[:, 0], label='G')
plt.plot(losses_array[:, 1], label='F')
plt.plot(losses_array[:, 2], label='D_X')
plt.plot(losses_array[:, 3], label='D_Y')
plt.legend()
plt.savefig(f'{dir}/loss.png')

# Save the model
G.save(f'{dir}/G')
F.save(f'{dir}/F')