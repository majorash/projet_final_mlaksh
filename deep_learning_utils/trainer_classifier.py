import numpy as np
import tensorflow as tf
import pandas as pd
from glob import glob
import json

import matplotlib
matplotlib.use("Agg")
from tensorflow.keras.applications import ResNet50, ResNet50V2, InceptionV3, MobileNetV2, VGG19, Xception, MobileNetV2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
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

batch_size = 64
nb_epochs = 5
dtrain=pd.read_csv("CheXpert-v1.0-small/train.csv")
dtrain = dtrain.fillna(0)


dtrain = dtrain[dtrain["Frontal/Lateral"] != "Lateral"]
dtrain = dtrain.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# print(dtrain.shape)
dtrain.describe().transpose()

# dealing with uncertanty (-1) values
dtrain = dtrain.replace(-1,0)

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

labels = list(dtrain.columns[1:])
train_generator = datagen.flow_from_dataframe(directory = './',
                                            dataframe = dtrain,
                                            class_mode = 'raw',
                                            x_col = "Path",
                                            y_col = labels,
                                            target_size = (224,224),
                                            batch_size = batch_size,
                                            seed = 43,
                                            subset = 'training')
val_generator = datagen.flow_from_dataframe(directory = './',
                                            dataframe = dtrain,
                                            class_mode = 'raw',
                                            x_col = "Path",
                                            y_col = labels,
                                            target_size = (224,224),
                                            batch_size = batch_size,
                                            seed = 43,
                                            subset = 'validation')

base_model = MobileNetV2(include_top = False, weights='imagenet')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer
predictions = Dense(14, activation='sigmoid')(x)


# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers[:-2]:
   layer.trainable = False
model.summary()
adam = tf.keras.optimizers.Adadelta()

adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer= adam, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(),'accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator,
    validation_steps = val_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [checkpoint])
with open('file5epmnet.json', 'w') as f:
    json.dump(history.history, f)
model.save("5_epoch_mnet.h5")
