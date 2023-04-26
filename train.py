import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re
import imageio
import sys

from util import preprocess_image_train
from models import build_generator, build_discriminator, CycleGAN
from callbacks import GenSameImg
from loss import DiscriminatorLoss, GeneratorLoss, CycleLoss, IdentityLoss

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

IMG_DIM = 256
DATASET = sys.argv[1]
CYCLEGAN_DATASETS = ['apple2orange', 'summer2winter_yosemite', 'horse2zebra', 'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo', 'maps', 'cityscapes', 'facades']

# Load dataset from cyclegan if it exists, otherwise load from local directory
if DATASET in CYCLEGAN_DATASETS:
    ds, ds_info = tfds.load(f'cycle_gan/{DATASET}', with_info=True, as_supervised=True)

    ds_train_A = ds['trainA']
    ds_train_B = ds['trainB']
    
    ds_train_A = ds_train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainA'].num_examples).batch(1)
    ds_train_B = ds_train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['trainB'].num_examples).batch(1)
elif os.path.exists(f'data/{DATASET}'):
    ds_train_A = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/trainA', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    ds_train_B = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/trainB', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    
    ds_train_A = ds_train_A.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
    ds_train_B = ds_train_B.map(preprocess_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
else:
    print(f'No dataset found for {DATASET}')
    exit()

# Initialize generators and discriminators
G = build_generator(IMG_DIM)
F = build_generator(IMG_DIM)
D_X = build_discriminator(IMG_DIM)
D_Y = build_discriminator(IMG_DIM)

# Initialize optimizers
G_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
F_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
D_X_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
D_Y_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# Create a directory to save the results
i = 0
while f'{DATASET}_{i}' in os.listdir('results'):
    i += 1
results_dir = f'results/{DATASET}_{i}'
os.mkdir(results_dir)
os.mkdir(f'{results_dir}/img')

# Train the model
lambd = 10
cycleGAN = CycleGAN(G, F, D_X, D_Y)
cycleGAN.compile(G_opt, F_opt, D_X_opt, D_Y_opt, GeneratorLoss(), DiscriminatorLoss(), CycleLoss(lambd), IdentityLoss(lambd))
hist = cycleGAN.fit(tf.data.Dataset.zip((ds_train_A, ds_train_B)), epochs=200, verbose=2,
                    callbacks=[
                        GenSameImg(f'tests/{DATASET}', results_dir),
                        tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0002 if epoch < 100 else 0.0002 - 0.0002 * (epoch - 100) / 100)
                    ])

# Plot the loss
plt.figure()
plt.plot(hist.history['G_loss'], label='G_loss')
plt.plot(hist.history['F_loss'], label='F_loss')
plt.plot(hist.history['D_X_loss'], label='D_X_loss')
plt.plot(hist.history['D_Y_loss'], label='D_Y_loss')
plt.legend()
plt.savefig(f'{results_dir}/loss.png')

# Save the model
G.save(f'{results_dir}/G')
F.save(f'{results_dir}/F')

# Make a video of the training process
frames = []
imgs = sorted(os.listdir(f'{results_dir}/img'), key=lambda x: int(re.sub('\D', '', x)))
for img in imgs:
    img_array = plt.imread(f'{results_dir}/img/{img}') * 255
    frames.append(img_array.astype('uint8'))

with imageio.get_writer(f'{results_dir}/train.gif', mode='I') as writer:
    for frame in frames:
        writer.append_data(frame)