import tensorflow as tf
import tensorflow_datasets as tfds

# Load the Food101 dataset (train split)
train_data, ds_info = tfds.load(name="food101", split="train", shuffle_files=True, as_supervised=True, with_info=True)

def preprocess_image(image, label):
  image = tf.image.resize(image, size=(128, 128))
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

batch_size = 32

# Filter out non-square images
train_data = train_data.filter(lambda image, label: tf.equal(tf.shape(image)[0], tf.shape(image)[1]))
train_data = train_data.map(preprocess_image)

import matplotlib.pyplot as plt

# Define a function to show images from the dataset
def show_images(dataset, num_images=5):
    # Get an iterator over the dataset
    iterator = iter(dataset)
    
    # Plot the images
    for i in range(num_images):
        image, label = next(iterator)
        plt.figure(figsize=(3, 3))
        plt.imshow(image.numpy())
        plt.axis('off')
        plt.show()

# Show some images from the preprocessed dataset
show_images(train_data, num_images=5)

batch_size = 64
# x_train = x_train.shuffle(batch_size)
train_data = train_data.batch(batch_size)
train_data = train_data.prefetch(tf.data.AUTOTUNE)