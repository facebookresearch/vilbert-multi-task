# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import os
import pdb
import numpy as np
import json
import sys

FIELDNAMES = [
    "image_id",
    "image_w",
    "image_h",
    "num_boxes",
    "boxes",
    "features",
    "cls_prob",
]
import csv
import base64

csv.field_size_limit(sys.maxsize)
import sys
import pickle
import lmdb  # install lmdb by "pip install lmdb"

count = 0
num_file = 1
name = "/srv/share2/jlu347/bottom-up-attention/feature/refcoco_unc/refcoco_unc_resnet101_faster_rcnn_genome.tsv.%d"
infiles = [name % i for i in range(num_file)]

save_path = os.path.join("refcoco_resnet101_faster_rcnn_genome.lmdb")
env = lmdb.open(save_path, map_size=1099511627776)

id_list = []
with env.begin(write=True) as txn:
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)
            for item in reader:
                img_id = str(item["image_id"]).encode()
                id_list.append(img_id)
                txn.put(img_id, pickle.dumps(item))

                if count % 1000 == 0:
                    print(count)
                count += 1
    txn.put("keys".encode(), pickle.dumps(id_list))

print(count)
