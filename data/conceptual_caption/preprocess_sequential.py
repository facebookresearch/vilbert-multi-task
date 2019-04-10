import os
import numpy as np
# from tensorpack.dataflow import RNGDataFlow, PrefetchDataZMQ
from tensorpack.dataflow import *
import lmdb
import json
import pdb

class Conceptual_Caption(RNGDataFlow):
    """
    """
    def __init__(self, corpus_path, shuffle=False):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.shuffle = shuffle
        self.num_file = 20
        self.envs = {}
        for i in range(self.num_file):
            imagePath = os.path.join(corpus_path, str(i)+'_new.lmdb')
            self.envs[i] = lmdb.open(imagePath,
                                    max_readers=1,
                                    readonly=True,
                                    lock=False,
                                    readahead=False,
                                    meminit=False)

        print('\nUse the soft label as the visual target...')
        self.target_envs = {}
        for i in range(self.num_file):
            imagePath = os.path.join(corpus_path, str(i)+'_pred.lmdb')
            self.target_envs[i] = lmdb.open(imagePath,
                                            max_readers=1,
                                            readonly=True,
                                            lock=False,
                                            readahead=False,
                                            meminit=False)

        print('\nloading caption file...', end=" ")
        self.caption_files = {}
        for i in range(self.num_file):
        # if True:
            print(i, end=" ")
            captionPath = os.path.join(corpus_path, str(i)+'_cap.json')
            self.caption_files[i] = json.load(open(captionPath, 'r'))

        self.num_per_file = [len(i) for i in self.caption_files.values()]
        self.num_caps = sum(self.num_per_file)
        # self.num_caps = 100
        print('\ntotal image captions in the dataset %d' %(self.num_caps))        

        print('\nloading location file...', end=" ")        
        self.location_files = {}
        for i in range(self.num_file):
            print(i, end=" ")
            locationPath = os.path.join(corpus_path, str(i)+'_loc_size.json')
            self.location_files[i] = json.load(open(locationPath, 'r'))

        self.hard_negative = {}
        for i in range(self.num_file):
            print(i, end=" ")
            hardnegativePath = os.path.join(corpus_path, str(i)+'_hard_negative.npy')
            self.hard_negative[i] = np.load(hardnegativePath)

        self.index_map = []
        # given an index, also create the map that give the file and associate index. 
        count = 0
        for i, caption_file in self.caption_files.items():
            for j, caption in enumerate(caption_file):
                self.index_map.append([i,j]) 

    def __len__(self):
        return self.num_caps

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            file_id, index_id = self.index_map[k]
            image_id = self.caption_files[file_id][index_id]['image_id']
            caption = self.caption_files[file_id][index_id]['caption']

            env = self.envs[file_id]
            with env.begin(write=False) as txn:
                image_feature =  np.frombuffer(txn.get(image_id.encode()), dtype='float32').reshape(36, 2048)

            # image_feature = image_feature.reshape(1, 36, 2048)

            target_env = self.target_envs[file_id]
            with target_env.begin(write=False) as txn:
                image_target = np.frombuffer(txn.get(image_id.encode()), dtype='float32').reshape(36, 1601)

            # image_target = image_target.reshape(1, 36, 1601)

            image_location = self.location_files[file_id][index_id]['location']
            width = self.location_files[file_id][index_id]['width']
            height = self.location_files[file_id][index_id]['height']
            im_scale = 1

            image_location = np.array(image_location)
            image_location[:,0] = image_location[:,0] / width / im_scale
            image_location[:,2] = image_location[:,2] / width / im_scale

            image_location[:,1] = image_location[:,1] / height / im_scale
            image_location[:,3] = image_location[:,3] / height / im_scale

            # image_location = image_location.reshape(1,36,4)
# 
            # hard_negative_list = []
            image_id = int(image_id)
            hard_negative_id = self.hard_negative[file_id][index_id]
            # hard_negative_id = hard_negative_id.reshape(1, 50)

            # pdb.set_trace()

            yield [image_feature, image_target, image_location, hard_negative_id, file_id, index_id]
            # yield image_feature

if __name__ == '__main__':
    corpus_path = '/coc/dataset/conceptual_caption/training'
    ds = Conceptual_Caption(corpus_path)

    # for data in ds:
        # pdb.set_trace()

    ds1 = PrefetchDataZMQ(ds, nr_proc=1)
    LMDBSerializer.save(ds1, 'training_feat_all.lmdb')
