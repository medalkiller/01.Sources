#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:09:22 2023

@author: julien
"""

from tensorflow import keras
from keras.models import load_model
import tensorflow as tf
import cv2

#import pandas as pd
import pandas as pd
import numpy as np

import tensorflow_hub as hub


import matplotlib.pyplot as plt
import matplotlib.image as mpimg


datagen_kwargs = dict(rescale=1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
test_generator = test_datagen.flow_from_directory(
    directory=r"./Dataset/test/",
    target_size=(28, 28),
    color_mode="grayscale",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


model = tf.keras.models.load_model('outputs/best_model.h5')


# y_sigmoid = model.predict(img_normalized[np.newaxis, ...])
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


labels = {'A':0, 'B': 1}
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})

results

num_images = len(results)
num_rows = 3
num_cols = int(np.ceil(len(results)/num_rows))

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
axes = axes.flatten()
# Loop through DataFrame rows and plot images
for i, (image_path, ax) in enumerate(zip(results["Filename"].to_list(), axes)):
    if i < num_images:
        img = mpimg.imread("./Dataset/test/"+results['Filename'][i])
        ax = axes[i] if num_cols > 1 else axes
        ax.imshow(img)
        ax.set_title(f"{results['Predictions'][i]} | {int(np.max(pred[i])*100)} %")
        ax.axis('off')
    else:
        ax.axis('off')  


# # Display legends
# unique_labels = results['Predictions'].unique()
# legend_labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='C{}'.format(idx), markersize=10, label=label) for idx, label in enumerate(unique_labels)]
# plt.legend(handles=legend_labels, loc='upper right')

plt.tight_layout()
plt.savefig("result.jpg", bbox_inches='tight')  # bbox_inches='tight' ensures no cropping
plt.show()
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)
# test_generator = train_datagen.flow_from_directory("Dataset/test/",shuffle=False,target_size=IMAGE_SHAPE)

# tf_model_predictions = model.predict(test_generator)
# y_pred    = np.argmax(tf_model_predictions, axis=-1)
