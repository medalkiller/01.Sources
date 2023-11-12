#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:53:20 2023

@author: julien
"""
import scipy

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import sys,os
from importlib import reload
import cv2

import tensorflow_hub as hub

CLASSES=['A', 'B']

TRAINING_DATA_DIR = str("Dataset/train/")
VALIDATION_DATA_DIR = str("Dataset/test/")
IMAGE_SHAPE = (28, 28) # (height, width) in no. of pixels

datagen_kwargs = dict(rescale=1./255)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
train_generator = train_datagen.flow_from_directory(TRAINING_DATA_DIR, shuffle=True,target_size=IMAGE_SHAPE, class_mode="categorical",color_mode='grayscale', )
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
valid_generator = valid_datagen.flow_from_directory(VALIDATION_DATA_DIR, shuffle=True,target_size=IMAGE_SHAPE, class_mode="categorical", color_mode='grayscale',)

model = keras.models.Sequential()

model.add( keras.layers.Input((IMAGE_SHAPE[0],IMAGE_SHAPE[1],1)) )

# # model.add( keras.layers.Conv2D(32, (3,3),  activation='relu') )
# # model.add( keras.layers.MaxPooling2D((2,2)))
# # model.add( keras.layers.Dropout(0.2))

# # model.add( keras.layers.Conv2D(64, (3,3),  activation='relu') )
# # model.add( keras.layers.MaxPooling2D((2,2)))
# # model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Conv2D(128, (3,3), activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)))
model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Conv2D(64, (3,3),  activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)))
model.add( keras.layers.Dropout(0.2))

model.add( keras.layers.Conv2D(32, (3,3),  activation='relu') )
model.add( keras.layers.MaxPooling2D((2,2)))
model.add( keras.layers.Dropout(0.2))


model.add( keras.layers.Flatten())
model.add( keras.layers.Dense(256, activation='relu'))
model.add( keras.layers.Dropout(0.5))

model.add( keras.layers.Dense(2, activation='softmax'))

model.summary()

bestmodel_callback = tf.keras.callbacks.ModelCheckpoint(filepath='outputs/best_model.h5', verbose=0, monitor='val_accuracy', save_best_only=True)

#model = tf.keras.Sequential([
  #hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",
  #output_shape=[1280],
  #trainable=False),
  #tf.keras.layers.Dropout(0.4),
  #tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
#])
#model.build([None, 224, 224, 3])
#model.summary()
optimizer = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epoch = np.ceil(train_generator.samples/train_generator.batch_size)

val_steps_per_epoch = np.ceil(valid_generator.samples/valid_generator.batch_size)

hist = model.fit(
 train_generator, 
 epochs=150,
 verbose=1,
 steps_per_epoch=steps_per_epoch,
 validation_steps=val_steps_per_epoch,
 validation_data = valid_generator,
 callbacks = [bestmodel_callback]
).history



model = tf.keras.models.load_model(
       ('outputs/best_model.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

final_loss, final_accuracy = model.evaluate(valid_generator, steps = val_steps_per_epoch)
print("Final loss: {:.2f}".format(final_loss))
print("Final accuracy: {:.2f}%".format(final_accuracy * 100))

# last_layer = model.layers[-3].output  # Get the output of the last convolutional layer
# feature_extraction_model = tf.keras.Model(inputs=model.input, outputs=last_layer)

# # Iterate through the batches of images in train_generator and plot feature maps
# for batch in train_generator:
#     images, _ = batch  # Assuming the generator yields batches of (images, labels)
#     feature_maps_batch = feature_extraction_model.predict(images)
    
#     # Iterate through each image in the batch and plot feature maps
#     for i in range(images.shape[0]):
#         plt.figure(figsize=(16, 16))
#         num_feature_maps = feature_maps_batch.shape[-1]
#         for j in range(num_feature_maps):
#             plt.subplot(8, 8, j + 1)
#             plt.imshow(feature_maps_batch[i, :, :, j], cmap='viridis')
#             plt.axis('off')
#         plt.show()

# img = cv2.imread("Dataset/cropped/Chene/IMG_20230806_080714.jpg")
# img = cv2.imread("Dataset/test/Chene/online1_chene.jpg")
# img = cv2.imread("Dataset/train/pin/IMG_20230806_080959_8.jpg")
# resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
# img_normalized = cv2.normalize(resized, None, 0, 1.0,cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# y_sigmoid = model.predict(img_normalized[np.newaxis, ...])
# y_pred    = np.argmax(y_sigmoid, axis=-1)
# print(CLASSES[y_pred[0]])

# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
# test_generator = train_datagen.flow_from_directory("Dataset/test/",shuffle=False,target_size=IMAGE_SHAPE)

# tf_model_predictions = model.predict(test_generator)
# y_pred    = np.argmax(tf_model_predictions, axis=-1)

# print("Prediction results shape:", tf_model_predictions.shape)
# plt.figure(figsize=(10,9))
# plt.subplots_adjust(hspace=0.5)
# for n in range((len(predicted_labels)-2)):
#  plt.subplot(6,5,n+1)
#  plt.imshow(val_image_batch[n])
#  color = "green" if predicted_ids[n] == true_label_ids[n] else "red"
#  plt.title(predicted_labels[n].title(), color=color)
#  plt.axis('off')
# _ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
