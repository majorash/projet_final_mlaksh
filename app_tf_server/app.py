import json

import numpy as np
import requests
import base64
import json
from io import BytesIO

import numpy as np
import logging
import requests
import sys
from flask import Flask, request, jsonify
from PIL import Image
from scipy.spatial.distance import cosine
from flaskext.mysql import MySQL
import pandas as pd
# from flask_cors import CORS

app = Flask(__name__)
mysql = MySQL()
app.secret_key = 'javascript is a spook - Max Stirner'

app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'password'
app.config['MYSQL_DATABASE_DB'] = 'RadioApp'
app.config['MYSQL_DATABASE_HOST'] = 'db'
mysql.init_app(app)

# Uncomment this line if you are making a Cross domain request
# CORS(app)


@app.route('/vectorExtract/infer', methods=['POST'])
def vector_extractor():
    # Decoding and pre-processing base64 image
    img_path = request.form['image_path']
    img = Image.open(img_path).resize((224,224)).convert('RGB')
    

    data = np.array(img)/255
    data = data.reshape(1,224,224,3)
    # this line is added because of a bug in tf_serving(1.10.0-dev)
    # img = img.astype('float16')
    
    # Creating payload for TensorFlow serving request
    payload = {
        "instances": data.tolist()
    }

    # Making POST request
    r = requests.post('http://tf_server:8501/v1/models/feature_vec:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))


    # Returning JSON response to the frontend
    return jsonify(pred)


@app.route('/classifier/predict', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img_path = request.form['image_path']
    img = Image.open(img_path).resize((224,224)).convert('RGB')
    data = np.array(img)/255
    print(data.shape, file=sys.stderr)
    data = data.reshape(1,224,224,3)
    
    # Creating payload for TensorFlow serving request
    payload = {
    "instances": data.tolist()
    }

    # Making POST request
    r = requests.post('http://tf_server:8501/v1/models/classification:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(pred)


@app.route('/cosine_prox', methods=['POST'])
def search():
    # Decoding and pre-processing base64 image
    img_path = request.form['image_path']

    conn = mysql.connect()
    cursor = conn.cursor()
    query = ("SELECT latent_vector FROM images "
         "WHERE img_path = %s ")
    cursor.execute(query, img_path)
    target_vec = cursor.fetchone()
    target_vec_arr = np.frombuffer(target_vec[0])
    # print(data)
    conn.commit()
    cursor.close()
    conn.close()


    conn = mysql.connect()
    cursor = conn.cursor()
    query = ("SELECT latent_vector, img_path from images where img_path <> \"" + img_path +"\" ;")
    # cursor.callproc('sp_GetAllVecs')
    cursor.execute(query)
    all_vecs = cursor.fetchall()
    df = pd.DataFrame(all_vecs)
    # print(data)
    conn.commit()
    cursor.close()
    conn.close()
    df.columns = ('bytevec','path')
    df["vec"] = df['bytevec'].apply(np.frombuffer)
    df["prox"] = df["vec"].apply(lambda x: cosine(x,target_vec_arr))
    df = df.sort_values(by=['prox'])
    res = df[['path']]
    res = res.iloc[:6]
    
    res.reset_index(inplace = True, drop = True)
    print(list(res['path'].values), file=sys.stderr)
    list__ = list(res['path'].values)
    b64_list = list()
    for i in list(list__):
        img = Image.open(i).resize((224,224)).convert('RGB')
        buffer = BytesIO()
        img.save(buffer,format="JPEG")              #Enregistre l'image dans le buffer
        myimage = buffer.getvalue()                     
        b64 = "data:image/png;base64,"+base64.b64encode(myimage).decode('utf8')
        b64_list.append(b64)
    
    print(len(b64_list), file=sys.stderr)
    json_response = json.dumps(b64_list)
    # Returning JSON response to the frontend
    return json_response

if __name__ == '__main__':
    # model = ResNet50(weights='imagenet')
    app.run(debug=False, threaded=False,host = '0.0.0.0', port = 5001)