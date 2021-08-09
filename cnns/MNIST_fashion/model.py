import tensorflow as tf
import numpy as np
import logging
import os
from tensorflow import keras
import math
logger = tf.get_logger()
logger.setLevel(logging.ERROR)



#Building the model with Keras

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                                                          input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation=tf.nn.relu),
        tf.keras.layers.MaxPooling2D((2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])


#Compiling the model

model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(),  metrics = ['accuracy'])



#Grapping data from data_prep.py
from data_prep import *


#Training set up

BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                                                            save_weights_only=True,
                                                                                                  verbose=1)


model.fit(train_dataset, epochs = 3, steps_per_epoch = math.ceil(num_train_examples/BATCH_SIZE))
