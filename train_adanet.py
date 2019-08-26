'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  reference : https://github.com/tensorflow/adanet/blob/master/adanet/examples/tutorials/customizing_adanet.ipynb
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

import time
from utils import AdaEstimator, generator, preprocess_image
import numpy as np
import tensorflow as tf

from load_bq import load_test_bq, load_train_bq

"""
    If you want to load Dataset Google Big Query on real time, See load_bq.py.
    I had cached MNIST Dataset only first time.
"""

load_test_bq()
x_test = np.load('data/x_test.npy')
y_test = np.load('data/y_test.npy')

load_train_bq()
x_train = np.load('data/x_train.npy')
y_train = np.load('data/y_train.npy')

def input_fn(partition, training, batch_size, RANDOM_SEED):
    """Generate an input_fn for the Estimator."""
    def _input_fn():
        if partition == "train":
            dataset = tf.data.Dataset.from_generator(
                generator(x_train, y_train), (tf.float32, tf.int32), ((28 * 28), ()))
        else:
            dataset = tf.data.Dataset.from_generator(
                generator(x_test, y_test), (tf.float32, tf.int32), ((28 * 28), ()))

        if training:
            dataset = dataset.shuffle(10 * batch_size, seed=RANDOM_SEED).repeat()

        dataset = dataset.map(preprocess_image).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def serving_input_receiver_fn():
    inputs = {
        "images": tf.placeholder(tf.float32, [None, 28, 28, 1], name="AKAKAKAK"),
    }
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def logging(logfile, results):
    with open(logfile, "a") as f:
        print('Finished Training')
        print("Accuracy: %f\n" % results["accuracy"])
        print("Loss: %f\n\n" % results["average_loss"])
        f.write("Accuracy: %f\n" % results["accuracy"])
        f.write("Loss: %f\n\n" % results["average_loss"])

def main(args):

    estim = AdaEstimator(args=args, NUM_CLASSES=10,
                         input_fn=input_fn)
    estimator = estim.get_estimator()

    results, _  = tf.estimator.train_and_evaluate(
        estimator,
        train_spec=tf.estimator.TrainSpec(
            input_fn=input_fn("train", training=True,
                              batch_size=args.BATCH_SIZE,
                              RANDOM_SEED=args.RANDOM_SEED),
            max_steps=args.TRAIN_STEPS),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=input_fn("test", training=False,
                              batch_size=args.BATCH_SIZE,
                              RANDOM_SEED=args.RANDOM_SEED),
            steps=None))

    estimator.export_savedmodel(args.SAVED_DIR,
        serving_input_receiver_fn=serving_input_receiver_fn)

    logging(logfile=args.logfile, results=results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--RANDOM_SEED', help='randome seed', default=3)
    parser.add_argument('--NUM_CLASSES', help='number of classes', default=10)
    parser.add_argument('--TRAIN_STEPS', help='train steps', default=5000)
    parser.add_argument('--BATCH_SIZE', help='batch size', default=32)
    parser.add_argument('--LEARNING_RATE', help='learning reate', default=0.001)
    parser.add_argument('--ADANET_ITERATIONS', help='adanet iterations', default=2)
    parser.add_argument('--MODEL_DIR', help='customizing models',
                        default='./models')
    parser.add_argument('--SAVED_DIR', help='saved models directory',
                        default='./saved_model')
    parser.add_argument('--logfile', help='location of log file', default='log.txt')

    known_args, pipeline_args = parser.parse_known_args()
    print(known_args)
    main(args=known_args)
