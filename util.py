import tensorflow as tf
import numpy as np

def resize_and_normalize(img: tf.Tensor, img_dim: int = 256):
    img = tf.image.resize(img, (img_dim, img_dim))
    img = img / 127.5 - 1 # normalize to [-1, 1]
    return img

@tf.function()
def random_jitter(img: tf.Tensor):
    img_dim = img.shape[0]
    img = tf.image.resize(img, [img_dim + img_dim // 10, img_dim + img_dim // 10], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # add 10% to the size
    img = tf.image.random_crop(img, size=(img_dim, img_dim, 3)) # random crop back to the original size
    if tf.random.uniform(()) > 0.5: # 50% chance to flip the image
        img = tf.image.flip_left_right(img)
    return img

def preprocess_image_train(img, label):
    img = resize_and_normalize(img)
    img = random_jitter(img)
    return img

def preprocess_image_test(img, label):
    img = resize_and_normalize(img)
    return img

# Generate an image given a generator
def generate_img(generator, test_img: np.ndarray):
    prediction = generator(np.expand_dims(test_img, axis=0), training=False).numpy()[0]
    img = np.concatenate((test_img, prediction), axis=1)
    return img * 0.5 + 0.5