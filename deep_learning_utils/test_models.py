import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential

from tensorflow.keras.applications import ResNet50, ResNet50V2, InceptionV3, MobileNetV2, VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from sklearn.metrics import classification_report
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
batch_size = 16
dtest = pd.read_csv("CheXpert-v1.0-small/valid.csv")
dtest = dtest.fillna(0)


dtest = dtest.drop(["Sex", "Age", "Frontal/Lateral", "AP/PA"], axis=1)

# print(dtrain.shape)
dtest.describe().transpose()

# dealing with uncertanty (-1) values
dtest = dtest.replace(-1,0)
test_datagen=ImageDataGenerator(rescale=1./255.)
labels = list(dtest.columns[1:])
test_generator = test_datagen.flow_from_dataframe(directory = './',
                                            dataframe = dtest,
                                            class_mode = 'raw',
                                            x_col = "Path",
                                            y_col = labels,
                                            target_size = (224,224),
                                            batch_size = batch_size,
                                            seed = 43)

model_F = load_model('model_inceptionv3_Balanced.h5')

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

# prediction and performance assessment
test_generator.reset()
pred=model_F.predict_generator(test_generator, steps=STEP_SIZE_TEST)
pred_bool = (pred >= 0.5)

y_pred = np.array(pred_bool,dtype =int)[:200]

y_true = np.array(dtest.iloc[:,1:15],dtype=int)[:200]
print(y_true.shape)
print(y_pred.shape)

print(classification_report(y_true, y_pred,target_names=list(dtest.columns[1:15])))

acc = model_F.evaluate(test_generator, steps=STEP_SIZE_TEST)

print('Test accuracy:', acc)
