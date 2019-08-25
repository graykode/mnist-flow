'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  Unit Test for Google query to numpy array
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from google.cloud import bigquery
client = bigquery.Client()

def load_test_bq():
    if not os.path.exists('data/x_test.npy'):
        test_query = ("SELECT image, label FROM mnist.test ORDER BY RAND()")
        test_query_job = client.query(
            test_query,
        )

        x_test = []
        y_test = []

        for row in test_query_job:
            image = np.asarray(row.image.split(',')).astype(np.uint8)
            label = np.asarray(row.label).astype(np.uint8)

            x_test.append(image)
            y_test.append(label)

        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        np.save('data/x_test', x_test)
        np.save('data/y_test', y_test)

def load_train_bq():
    if not os.path.exists('data/x_train.npy'):
        train_query = ("SELECT image, label FROM mnist.train ORDER BY RAND()")
        train_query_job = client.query(
            train_query,
        )

        x_train = []
        y_train = []

        for row in train_query_job:
            image = np.asarray(row.image.split(',')).astype(np.uint8)
            label = np.asarray(row.label).astype(np.uint8)

            x_train.append(image)
            y_train.append(label)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        np.save('data/x_train', x_train)
        np.save('data/y_train', y_train)
