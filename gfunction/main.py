'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  reference : https://github.com/ryfeus/gcf-packs/blob/master/tensorflow2.0/example/main.py
'''
from google.cloud import storage
import numpy as np
import cv2
from tensorflow.contrib import predictor
import os

# We keep model as global varibale so we don't have to reload it in case of warm invocations
model = None
MODEL_TAG = '1566790420'

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    os.mkdir(os.path.join('/tmp', MODEL_TAG, 'variables'))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_or_name=bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_name)
    for blob in blobs:
        filename = os.path.basename(blob.name)
        print('Blob {} downloaded to {}.'.format(
            blob.name,
            os.path.join(destination_file_name, 'variables', filename)))
        if 'variables' in filename:
            blob.download_to_filename(os.path.join(destination_file_name, 'variables', filename))
        else:
            blob.download_to_filename(os.path.join(destination_file_name, filename))

def unitest_download():
    # Test folder download on Google Storage
    os.mkdir(os.path.join('/tmp', MODEL_TAG))
    download_blob('skilled-circle-250909-ml', 'saved_model/1566790420/', '/tmp/1566790420')

def handler(request):
    global model

    f = request.files.get('file')
    img = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = img.reshape([-1, 28, 28, 1])

    # Model load which only happens during cold starts
    if model is None:
        os.mkdir(os.path.join('/tmp', MODEL_TAG))
        download_blob('skilled-circle-250909-ml',
                      os.path.join('saved_model', MODEL_TAG),
                      os.path.join('/tmp', MODEL_TAG))

        model = predictor.from_saved_model(os.path.join('/tmp', MODEL_TAG),
                                           signature_def_key="predict")

    pred = model({'images': img})
    result = str(pred['class_ids'][0][0])

    return result
