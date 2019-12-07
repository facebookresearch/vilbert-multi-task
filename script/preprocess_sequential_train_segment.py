# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from tensorpack.dataflow import *
import lmdb
import json
import pdb
import csv
import sys
import pandas as pd
import zlib
import base64
import tensorpack.dataflow as td

csv.field_size_limit(sys.maxsize)


class ConceptCapLoaderTrain(RNGDataFlow):
    def __init__(self, num_split):

        lmdb_file = "/srv/share/vgoswami8/conceptual_captions/training_feat_all.lmdb"

        caption_path = "/srv/share/vgoswami8/conceptual_captions/caption_train.json"
        print("Loading from %s" % lmdb_file)
        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = int(len(ds) / num_split) + 1
        ds = td.PrefetchDataZMQ(ds, nr_proc=1)
        ds = td.FixedSizeData(ds, self.num_dataset, keep_state=True)
        self.ds = ds
        self.ds.reset_state()

    def __iter__(self):

        for batch in self.ds.get_data():
            yield batch

    def __len__(self):
        return self.ds.size()


if __name__ == "__main__":

    num_split = 8
    ds = ConceptCapLoaderTrain(num_split)
    for i in range(num_split):
        LMDBSerializer.save(
            ds, "data/conceptual_caption/training_feat_part_" + str(i) + ".lmdb"
        )
