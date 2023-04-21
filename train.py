import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from util import preprocess_image_train, preprocess_image_test
from models import build_generator, build_discriminator, CycleGAN
from callback import GenImg
from loss import DiscriminatorLoss, GeneratorLoss, CycleLoss, IdentityLoss

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMG_DIM = 256

image_set = 'apple2orange'
ds, ds_info = tfds.load(f'cycle_gan/{image_set}', with_info=True, as_supervised=True)

ds_train_A = ds['trainA']
ds_train_B = ds['trainB']

ds_test_A = ds['testA']
ds_test_B = ds['testB']

ds_train_A = ds_train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainA'].num_examples).batch(1)
ds_train_B = ds_train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainB'].num_examples).batch(1)
ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)

# Initialize generators and discriminators
G = build_generator(IMG_DIM)
F = build_generator(IMG_DIM)
D_X = build_discriminator(IMG_DIM)
D_Y = build_discriminator(IMG_DIM)

# Initialize optimizers
learning_rate = 0.0002
G_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
F_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
D_X_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
D_Y_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

# Train the model
lambd = 10
cycleGAN = CycleGAN(G, F, D_X, D_Y)
cycleGAN.compile(G_opt, F_opt, D_X_opt, D_Y_opt, GeneratorLoss(), DiscriminatorLoss(), CycleLoss(lambd), IdentityLoss(lambd))
hist = cycleGAN.fit(tf.data.Dataset.zip((ds_train_A, ds_train_B)), epochs=100, callbacks=[GenImg(ds_test_A, ds_test_B, image_set)])

# Plot the loss
plt.figure()
plt.plot(hist.history['G_loss'], label='G_loss')
plt.plot(hist.history['F_loss'], label='F_loss')
plt.plot(hist.history['D_X_loss'], label='D_X_loss')
plt.plot(hist.history['D_Y_loss'], label='D_Y_loss')
plt.legend()
plt.savefig(f'{dir}/loss.png')

# Save the model
G.save(f'{dir}/G')
F.save(f'{dir}/F')