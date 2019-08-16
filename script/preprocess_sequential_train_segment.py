import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
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
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(self):
    
        lmdb_file = "/coc/dataset/conceptual_caption/training_feat_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "data/conceptual_caption/training_feat_all.lmdb"
        
        caption_path = "data/conceptual_caption/caption_train.json"
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = int(len(ds) / 8) + 1

        # ds = td.PrefetchData(ds, 5000, 1)
        # ds = td.MapData(ds, preprocess_function)
        # # self.ds = td.PrefetchData(ds, 1)
        ds = td.PrefetchDataZMQ(ds, nr_proc=1)
        ds = td.FixedSizeData(ds, self.num_dataset, keep_state=True)    
        # self.ds = td.BatchData(ds, batch_size)
        self.ds = ds
        self.ds.reset_state()

    def __iter__(self):

        for batch in self.ds.get_data():
            yield batch

    def __len__(self):
        return self.ds.size()

if __name__ == '__main__':
    
    ds = ConceptCapLoaderTrain()
    for i in range(8):
        LMDBSerializer.save(ds, '/coc/dataset/conceptual_caption/training_feat_part_' + str(i) + 'new.lmdb')
