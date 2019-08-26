'''
  code by Tae Hwan Jung(Jeff Jung) @graykode
  reference : https://github.com/tensorflow/adanet/blob/master/adanet/examples/tutorials/customizing_adanet.ipynb
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import adanet
from adanet.examples import simple_dnn
import tensorflow as tf

def generator(images, labels):
    """Returns a generator that returns image-label pairs."""
    def _gen():
        for image, label in zip(images, labels):
            yield image, label
    return _gen

def preprocess_image(image, label=None):
    image = image / 255.
    image = tf.reshape(image, [28, 28, 1])
    features = {"images" : image}
    if label == None:
        return features
    return features, label

class AdaEstimator:
    def __init__(self, args=None,
                 NUM_CLASSES=10, input_fn=None):
        self.config = tf.estimator.RunConfig(
                save_checkpoints_steps=50000,
                save_summary_steps=50000,
                tf_random_seed=args.RANDOM_SEED,
            model_dir=args.MODEL_DIR)

        # We will average the losses in each mini-batch when computing gradients.
        self.RANDOM_SEED = args.RANDOM_SEED
        self.LEARNING_RATE = args.LEARNING_RATE
        self.ADANET_ITERATIONS = args.ADANET_ITERATIONS
        self.NUM_CLASSES = args.NUM_CLASSES
        self.BATCH_SIZE = args.BATCH_SIZE
        self.TRAIN_STEPS = args.TRAIN_STEPS
        self.input_fn = input_fn

        self.loss_reduction = tf.losses.Reduction.SUM_OVER_BATCH_SIZE
        # A `Head` instance defines the loss function and metrics for `Estimators`.
        self.head = tf.contrib.estimator.multi_class_head(
            NUM_CLASSES, loss_reduction=self.loss_reduction)

        # Some `Estimators` use feature columns for understanding their input features.
        self.feature_columns = [
            tf.feature_column.numeric_column("images", shape=[28, 28, 1])
        ]

    def get_estimator(self):
        estimator = adanet.Estimator(
            head=self.head,
            subnetwork_generator=simple_dnn.Generator(
                feature_columns=self.feature_columns,
                optimizer=tf.train.RMSPropOptimizer(learning_rate=self.LEARNING_RATE),
                seed=self.RANDOM_SEED),
            max_iteration_steps=self.TRAIN_STEPS // self.ADANET_ITERATIONS,
            evaluator=adanet.Evaluator(
                input_fn=self.input_fn("train", training=False,
                                  batch_size=self.BATCH_SIZE,
                                  RANDOM_SEED=self.RANDOM_SEED),
                steps=None),
            config=self.config)

        return estimator
