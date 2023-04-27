import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import os
from util import preprocess_image_test

IMG_DIM = 256
DATASET = sys.argv[1]
CYCLEGAN_DATASETS = ['apple2orange', 'summer2winter_yosemite', 'horse2zebra', 'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo', 'maps', 'facades']

# Load dataset from cyclegan if it exists, otherwise load from local directory
if DATASET in CYCLEGAN_DATASETS:
    ds, ds_info = tfds.load(f'cycle_gan/{DATASET}', with_info=True, as_supervised=True)

    ds_test_A = ds['testA']
    ds_test_B = ds['testB']
    
    ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
    ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)
elif os.path.exists(f'data/{DATASET}'):
    ds_test_A = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/testA', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    ds_test_B = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/testB', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    
    ds_train_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
    ds_train_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
else:
    print(f'No dataset found for {DATASET}')
    exit()

# show a few images
for i, img in enumerate(ds_test_A.take(10)):
    plt.imsave(f'tests/{DATASET}/A{i}.jpg', img[0].numpy() * 0.5 + 0.5)

for i, img in enumerate(ds_test_B.take(10)):
    plt.imsave(f'tests/{DATASET}/B{i}.jpg', img[0].numpy() * 0.5 + 0.5)