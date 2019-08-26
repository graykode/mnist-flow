'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import argparse
import numpy as np
import tensorflow as tf
import PIL.Image as img
from werkzeug import secure_filename
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import predictor
from flask import Flask, render_template, request

app = Flask(__name__)

SAVED_MODEL = ''

parser = argparse.ArgumentParser()
parser.add_argument('--SAVED_MODEL', help='Select saved model',
                    default='./saved_model/1566790420/')
args, _ = parser.parse_known_args()
predict_fn = predictor.from_saved_model(args.SAVED_MODEL, signature_def_key="predict")

def unittest():
    predict_fn = predictor.from_saved_model(args.SAVED_MODEL, signature_def_key="predict")
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    for i in range(10):
        pred = predict_fn({'images': x_test[i].reshape([-1, 28, 28, 1])})
        print(y_test[i], pred['class_ids'][0][0])

@app.route('/')
def index():
    # curl -v http://localhost:5000/
   return 'TEST'

@app.route('/upload', methods=['POST'])
def serving():
    # curl -F ‘file=@testdata/0.png’ http://localhost:5000/upload
    f = request.files.get('file')
    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = img.reshape([-1, 28, 28, 1])
    pred = predict_fn({'images': img})
    result = str(pred['class_ids'][0][0])
    return result

app.run(host='0.0.0.0', debug=True)

