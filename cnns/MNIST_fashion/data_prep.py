import tensorflow as tf
import numpy as np
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt


dataset, metadata = tfds.load('fashion_mnist', as_supervised = True, with_info = True)
def get_dataset(dataset, set):
    return dataset[set]

train_dataset = get_dataset(dataset, 'train')
test_dataset = get_dataset(dataset, 'test')

#Extracted class_names ( from offivial website )
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']


def normalize(images, labels):
        images = tf.cast(images, tf.float32)
        images /= 255
        return images, labels

train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)


train_dataset =  train_dataset.cache()
test_dataset  =  test_dataset.cache()



