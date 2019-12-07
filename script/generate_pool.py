# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import gc
import random
import pprint
from six.moves import range
from time import gmtime, strftime
from timeit import default_timer as timer
from sklearn.neighbors import KDTree, BallTree

import numpy as np
from multiprocessing.pool import ThreadPool as Pool
import time
import pymp
import _pickle as cPickle

import h5py
import numpy as np
import torch
import random
from six import iteritems
from six.moves import range
from sklearn.preprocessing import normalize
import json
import pdb
import jsonlines
from sklearn.neighbors import KDTree, BallTree


def get_neighbors(image_list, image_feature, kdt):

    total_image = len(image_list)
    batch_size = 1
    p_length = int(total_image / batch_size)
    # easy_pool = pymp.shared.array((total_image,100))
    hard_pool = pymp.shared.array((total_image, 100))

    with pymp.Parallel(40) as p:
        for index in p.range(0, p_length):
            ind = kdt.query(
                image_feature[index : index + 1], k=100, return_distance=False
            )
            hard_pool[index] = ind
            print("finish worker", index)

    return hard_pool


inputImg = "data/flick30k/flickr30k.h5"
inputJson = "data/flick30k/all_data_final_train_2014.jsonline"

with h5py.File(inputImg, "r") as features_h5:
    _image_ids = list(features_h5["image_ids"])

train_image_list = []
with jsonlines.open(inputJson) as reader:
    # Build an index which maps image id with a list of caption annotations.
    for annotation in reader:
        train_image_list.append(int(annotation["img_path"].split(".")[0]))

num_train = len(train_image_list)
print(len(train_image_list))

train_image_feature = pymp.shared.array([num_train, 2048])

with pymp.Parallel(40) as p:
    with h5py.File(inputImg, "r", libver="latest", swmr=True) as features_h5:
        for i in p.range(0, num_train):
            image_id = train_image_list[i]
            index = _image_ids.index(image_id)
            num_boxes = int(features_h5["num_boxes"][index])
            feature = features_h5["features"][index]
            train_image_feature[i] = feature[:num_boxes].sum(0) / num_boxes
            print("finish worker", i)

kdt = BallTree(train_image_feature[:, :], metric="euclidean")
print("finish create the ball tree")

train_hard_pool = get_neighbors(train_image_list, train_image_feature, kdt)

# save the pool info into
cache_root = "data/flick30k"
cache_file = os.path.join(cache_root, "hard_negative.pkl")
save_file = {}
save_file["train_hard_pool"] = train_hard_pool
save_file["train_image_list"] = train_image_list

cPickle.dump(save_file, open(cache_file, "wb"))
