import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import sys
import re
import os
from util import generate_img, preprocess_image_test

IMG_DIM = 256
NUM_IMG = int(sys.argv[2])

DATASET_DIR = sys.argv[1]
DATASET = re.match(r'([\w2]+).*', DATASET_DIR).group(1)
CYCLEGAN_DATASETS = ['apple2orange', 'summer2winter_yosemite', 'horse2zebra', 'monet2photo', 'cezanne2photo', 'ukiyoe2photo', 'vangogh2photo', 'facades']

# Load dataset from cyclegan if it exists, otherwise load from local directory
if DATASET in CYCLEGAN_DATASETS:
    ds, ds_info = tfds.load(f'cycle_gan/{DATASET}', with_info=True, as_supervised=True)

    ds_test_A = ds['testA'].map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testA'].num_examples).batch(1)
    ds_test_B = ds['testB'].map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(ds_info.splits['testB'].num_examples).batch(1)
elif os.path.exists(f'data/{DATASET}'):
    ds_test_A = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/testA', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    ds_test_B = tf.keras.preprocessing.image_dataset_from_directory(f'data/{DATASET}/testB', image_size=(IMG_DIM, IMG_DIM), batch_size=None, label_mode=None)
    
    ds_test_A = ds_test_A.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
    ds_test_B = ds_test_B.map(preprocess_image_test, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1)
else:
    print(f'No dataset found for {DATASET}')
    exit()

G = tf.saved_model.load(f'results/{DATASET_DIR}/G')
F = tf.saved_model.load(f'results/{DATASET_DIR}/F')

# test_A = plt.imread(f'tests/{DATASET}/A.jpg')
# test_A = resize_and_normalize(test_A)
# gen_A = generate_img(G, test_A)

# test_B = plt.imread(f'tests/{DATASET}/B.jpg')
# test_B = resize_and_normalize(test_B)
# gen_B = generate_img(F, test_B)
# plt.imsave(f'tests/gen_{DATASET}.jpg', np.concatenate((gen_A, gen_B)))

os.mkdir(f'results/{DATASET_DIR}/tests')
os.mkdir(f'results/{DATASET_DIR}/tests/A')
os.mkdir(f'results/{DATASET_DIR}/tests/B')

iter_A = iter(ds_test_A)
iter_B = iter(ds_test_B)

for i in range(NUM_IMG):
    test_A = next(iter_A)[0]
    test_B = next(iter_B)[0]
    gen_A = generate_img(G, test_A, single_img=True)
    gen_B = generate_img(F, test_B, single_img=True)
    plt.imsave(f'results/{DATASET_DIR}/tests/A/{i}.jpg', gen_A)
    plt.imsave(f'results/{DATASET_DIR}/tests/B/{i}.jpg', gen_B)