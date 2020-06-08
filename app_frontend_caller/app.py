import sys
import os
import glob
import re
import numpy as np
# import tensorflow as tf

# Keras
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.densenet import DenseNet121

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from flaskext.mysql import MySQL
from werkzeug.utils import secure_filename
import argparse
import base64
import json
import sys
import requests


# Define a flask app
app = Flask(__name__)
mysql = MySQL()
app.secret_key = 'javascript is a spook - Max Stirner'

app.config['MYSQL_DATABASE_PASSWORD'] = 'password'
app.config['MYSQL_DATABASE_DB'] = 'RadioApp'
app.config['MYSQL_DATABASE_HOST'] = 'db'
app.config['MYSQL_DATABASE_PORT'] = 3306
mysql.init_app(app)

# # Model saved with Keras model.save()
# weights = 'models/chestXnet.h5'

# # Load your trained model
# model = DenseNet121(weights = weights,classes = 14)
# # model._make_predict_function()          # Necessary
# # plot_model(model, to_file="model.png")
# model.summary()
print('Model loading...')
# You can also use pretrained model from Keras
# Check https://keras.io/applications/

# model = ResNet50(weights='imagenet')
#graph = tf.get_default_graph()

# print('Model loaded. Started serving...')

# print('Model loaded. Check http://127.0.0.1:5000/')




@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']
        # Save the file to ./uploads
        file_path = os.path.join(
            '/var/uploads', secure_filename(f.filename))
        f.save(file_path)
        _user_id = 'test'

        f = request.files['image']
        
        file_path = os.path.join(
            '/var/uploads', secure_filename(f.filename))


        API_VEC_ENDPOINT = "http://app_back:5001/vectorExtract/infer"
        API_CLASS_ENDPOINT = "http://app_back:5001//classifier/predict"
        # data to be sent to api
        data = {'image_path': file_path}

        # sending post request and saving response as response object
        r_vec = requests.post(url=API_VEC_ENDPOINT, data=data)
        r_class = requests.post(url=API_CLASS_ENDPOINT, data=data)
        
        
        json_r_vec = json.loads(r_vec.text)
        json_r_class = json.loads(r_class.text)

        class_pred = np.array(json_r_class["predictions"])
        class_list = ['No Finding', 'Enlarged Cardiomediastinum', 
                      'Cardiomegaly', 'Lung Opacity', 'Lung Lesion',
                      'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                      'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
                      'Fracture', 'Support Devices']
        index = np.argmax(class_pred)
        classs = class_list[index]
        latent_vector = np.array(json_r_vec["predictions"]).tobytes()


        conn = mysql.connect()
        cursor = conn.cursor()

        cursor.callproc('sp_addImage',(file_path,_user_id, classs,latent_vector))

        data = cursor.fetchall()
        # print(data)
        conn.commit()
        cursor.close()
        conn.close()
        
        return ("{}".format(classs))
    return None

@app.route('/reverse_search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, '/var/uploads', secure_filename(f.filename))
        f.save(file_path)
        print("processing4...", file=sys.stderr)
        API_SEARCH_ENDPOINT = "http://app_back:5001/cosine_prox"

        data = {'image_path': file_path}
        

        r_search = requests.post(url=API_SEARCH_ENDPOINT, data=data)
        print("processingpass...", file=sys.stderr)

        return ("{}".format(r_search.text))
    print("processingfail...", file=sys.stderr)
    return None

if __name__ == '__main__':
    # model = ResNet50(weights='imagenet')
    app.run(debug=False, threaded=False,host = '0.0.0.0', port = 5000)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
