#!/usr/bin/python

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

with open('data/train.txt', 'w') as output:
  for num, (image, label) in enumerate(zip(x_train, y_train)):
    output.write('%d:%s:%d' % (num, ','.join(map(str, image.flatten().tolist())), label))
    output.write('\n')

with open('data/test.txt', 'w') as output:
  for num, (image, label) in enumerate(zip(x_test, y_test)):
    output.write('%d:%s:%d' % (num, ','.join(map(str, image.flatten().tolist())), label))
    output.write('\n')
