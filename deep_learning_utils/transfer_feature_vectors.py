from tensorflow.keras.applications import MobileNetV2, ResNet50V2, InceptionV3
from tensorflow.keras.models import Model

import numpy as np
import tensorflow as tf
import pandas as pd
from glob import glob
import json

import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, Reshape, GlobalMaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image

growth = True
if growth:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

batch_size = 128
nb_epochs = 30
dtrain=pd.read_csv("CheXpert-v1.0-small/train.csv")
dtrain = dtrain.fillna(0)
dtrain = dtrain[dtrain["Frontal/Lateral"] != "Lateral"]

dtrain = dtrain.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# print(dtrain.shape)
dtrain.describe().transpose()

# dealing with uncertanty (-1) values
dtrain = dtrain.replace(-1,1)

# dtest = pd.read_csv("CheXpert-v1.0-small/test.csv")
# dval = dval.fillna(0)
#
#
# dtest = dval.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
#
# # print(dtrain.shape)
# dtest.describe().transpose()
#
# # dealing with uncertanty (-1) values
# dtest = dval.replace(-1,1)

datagen=ImageDataGenerator(rescale=1./255,validation_split = 0.2)

dtest = pd.read_csv("CheXpert-v1.0-small/valid.csv")
dtest = dtest.fillna(0)


dtest = dtest.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# print(dtrain.shape)
dtest.describe().transpose()

# dealing with uncertanty (-1) values
dtest = dtest.replace(-1,1)

Mnet = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))
output = Mnet.layers[-1].output
output = GlobalMaxPooling2D()(output)
output = tf.keras.layers.Flatten()(output)
Mnet = Model(Mnet.input, output)
for layer in Mnet.layers:
    layer.trainable = False

Mnet.summary()
Mnet.save("featurevec_inf.h5")
dtest = pd.read_csv("CheXpert-v1.0-small/valid.csv")
dtest = dtest.fillna(0)


dtest = dtest.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)
# print(dtrain.shape)
dtest.describe().transpose()
paths = dtest.iloc[:,0].values
# dealing with uncertanty (-1) values
dtest = dtest.replace(-1,1)
original = image.load_img(paths[0], target_size=(224, 224))

# Preprocessing the image

# Convert the PIL image to a numpy array
# IN PIL - image is in (width, height, channel)
# In Numpy - image is in (height, width, channel)
numpy_image = image.img_to_array(original)

# Convert the image / images into batch format
# expand_dims will add an extra dimension to the data at a particular axis
# We want the input matrix to the network to be of the form (batchsize, height, width, channels)
# Thus we add the extra dimension to the axis 0.
numpy_image = numpy_image / 255
image_batch = np.expand_dims(numpy_image, axis=0)
print(np.array(image_batch).shape)
pred = Mnet.predict(image_batch)

print(pred.shape)