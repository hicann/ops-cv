"""
Copyright 2023 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import logging
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op, dtypes
from tensorflow.python.ops import gen_image_ops
import random
# prama1: file_name: the file which store the data
# param2: data: data which will be stored
# param3: fmt: format
def write_file_txt(file_name, data, fmt="%s"):
    if (file_name is None):
        print("file name is none, do not write data to file")
        return
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    np.savetxt(file_name, data.flatten(), fmt=fmt, delimiter='', newline='\n')
# prama1: file_name: the file which store the data
# param2: dtype: data type
# param3: delim: delimiter which is used to split data
def read_file_txt(file_name, dtype, delim=None):
    return np.loadtxt(file_name, dtype=dtype, delimiter=delim).flatten()
# prama1: file_name: the file which store the data
# param2: delim: delimiter which is used to split data
def read_file_txt_to_bool(file_name, delim=None):
    in_data = np.loadtxt(file_name, dtype=str, delimiter=delim)
    bool_data = []
    for item in in_data:
        if item == "False":
            bool_data.append(False)
        else:
            bool_data.append(True)
    return np.array(bool_data)
# prama1: data_file: the file which store the generation data
# param2: shape: data shape
# param3: dtype: data type
# param4: rand_type: the method of generate data, select from "randint, uniform"
# param5: data lower limit
# param6: data upper limit
def gen_data_file(data_file, shape, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = np.random.randint(low, high, size=shape)
    elif rand_type == "uniform":
        rand_data = np.random.uniform(low, high, size=shape)
    elif rand_type == "complex":
        r1 = np.random.uniform(low, high, size=shape)
        r2 = np.random.uniform(low, high, size=shape)
        rand_data = np.empty((shape[0], shape[1], shape[2]), dtype=dtype)
        for i in range(0, shape[0]):
            for p in range(0, shape[1]):
                for k in range(0, shape[2]):
                    rand_data[i, p, k] = complex(r1[i, p, k], r2[i, p, k])
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data

def gen_data_file2(data_file, dtype, rand_type, low, high):
    if rand_type == "randint":
        rand_data = random.randint(low, high)
    else:
        rand_data = random.uniform(low, high)
    data = np.array(rand_data, dtype=dtype)
    write_file_txt(data_file, data, fmt="%s")
    return data

def config(execute_type):
    if execute_type == 'cpu':
        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
    return session_config

def gen_random_data_fp16_1():
    data_files=["data/non_max_suppression_v3_data_boxes_fp16_1.txt",
                 "data/non_max_suppression_v3_data_scores_fp16_1.txt",
                 "data/non_max_suppression_v3_data_output_int32_1.txt"]
    tf.compat.v1.disable_eager_execution()
    np.random.seed(3457)
    shape_boxes_data = [4,4]
    shape_scores_data = [4]
    a = gen_data_file(data_files[0],shape_boxes_data,np.float16,"randint",1.0,100.0)
    b = gen_data_file(data_files[1],shape_scores_data,np.float16,"randint",1.0,100.0)
    boxes_data = tf.compat.v1.placeholder(tf.float16,shape=shape_boxes_data)
    scores_data = tf.compat.v1.placeholder(tf.float16,shape=shape_scores_data)
    max_output_size = 4
    iou_threshold = 1.0
    score_threshold = 0.5
    ret = gen_image_ops.non_max_suppression_v3(
        boxes = boxes_data,
        scores = scores_data,
        max_output_size = max_output_size,
        iou_threshold = iou_threshold,
        score_threshold = score_threshold)
    with tf.compat.v1.Session(config=config('cpu')) as session:
        data = session.run(ret, feed_dict={boxes_data:a, scores_data:b})
        print("non_max_suppression_v3 gen_random_data_fp16_1:", data.size)
    write_file_txt(data_files[2],data,fmt="%s")

def run():
    gen_random_data_fp16_1()

if __name__ == '__main__':
    run()