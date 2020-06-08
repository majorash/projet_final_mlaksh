import tensorflow as tf 
from tensorflow.keras.models import load_model

model1 = load_model('50_epoch_DenseNet.h5')

model1.save('savedmodel1/classification')
# model2 = load_model("featurevec_inf.h5")


# model2.save('savedmodel2/feature_vec')