import numpy as np
import tensorflow as tf
import pandas as pd
from glob import glob
import json

import matplotlib
matplotlib.use("Agg")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Conv2DTranspose, Reshape
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

labels = list(dtrain.columns[1:])
train_generator = datagen.flow_from_dataframe(directory = './',
                                            dataframe = dtrain,
                                            class_mode = 'input',
                                            x_col = "Path",
                                            target_size = (224,224),
                                            batch_size = batch_size,
                                            seed = 43,
                                            subset = 'training')
val_generator = datagen.flow_from_dataframe(directory = './',
                                            dataframe = dtrain,
                                            class_mode = 'input',
                                            x_col = "Path",
                                            target_size = (224,224),
                                            batch_size = batch_size,
                                            seed = 43,
                                            subset = 'validation')

# network parameters
image_size = 224
input_shape = (image_size, image_size, 3)
kernel_size = 5
latent_dim = 100
# encoder/decoder number of CNN layers and filters per layer
layer_filters = [16, 32, 64, 128, 256]

# build the autoencoder model
# first build the encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs

# stack of Conv2D(32)-Conv2D(64)
for filters in layer_filters:
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=2,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPool2D()(x)
    x = Dropout(0.3)(x)
# shape info needed to build decoder model so we don't do hand computation
# the input to the decoder's first Conv2DTranspose will have this shape
# shape is (7, 7, 64) which can be processed by the decoder back to (28, 28, 1)
shape = K.int_shape(x)

# generate the latent vector
x = Flatten()(x)
latent = Dense(latent_dim, name='latent_vector')(x)

# instantiate encoder model
encoder = Model(inputs, latent, name='encoder')
encoder.summary()

# build the decoder model
latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
# use the shape (7, 7, 64) that was earlier saved
x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
# from vector to suitable shape for transposed conv
x = Reshape((shape[1], shape[2], shape[3]))(x)

# stack of Conv2DTranspose(64)-Conv2DTranspose(32)
for filters in layer_filters[::-1]:
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPool2D()(x)
    x = Dropout(0.3)(x)
# reconstruct the denoised input
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()


# autoencoder = encoder + decoder
# instantiate autoencoder model
autoencoder = Model(inputs, decoder(encoder(inputs)), name='autoencoder')


autoencoder.compile('adadelta',loss="binary_crossentropy", metrics = ['accuracy'])
autoencoder.summary()
# Mean Square Error (MSE) loss function, Adam optimizer
decoder.summary()
chk = True
if chk:
    filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
else:
    checkpoint = None


history = autoencoder.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = val_generator,
    validation_steps = val_generator.samples // batch_size,
    epochs = nb_epochs,
    callbacks = [checkpoint])
# Get the dictionary containing each metric and the loss for each epoch
history_dict = history.history
with open('file.json', 'w') as f:
    json.dump(history_dict, f)
model.save("30epoch_autoencoder.h5")
