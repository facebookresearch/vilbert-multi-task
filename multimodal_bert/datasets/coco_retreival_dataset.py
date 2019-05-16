import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_annotations(annotations_jsonpath, data_split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""

    annotations_json = json.load(open(annotations_jsonpath))

    # Build an index which maps image id with a list of caption annotations.
    entries = []
    imgid2entry = {}
    count = 0
    for annotation in annotations_json["images"]:
        image_id = annotation['cocoid']
        split = annotation['split']
        imgid2entry[image_id] = []
        if split in data_split:
            for sentences in annotation['sentences']:
                entries.append(
                    {"caption": sentences["raw"], 'image_id':image_id}
                )
                imgid2entry[image_id].append(count)
                count += 1

    return entries, imgid2entry


class COCORetreivalDataset(Dataset):
    def __init__(
        self,
        name: str,
        annotations_jsonpath: str,
        image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_caption_length: int = 20,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        if name == 'train':
            data_split = ['train', 'restval']
        elif name == 'val':
            data_split = ['val']
        elif name == 'test':
            data_split == ['test']

        self._entries, self.imgid2entry = _load_annotations(annotations_jsonpath, data_split)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_caption_length

        image_info = cPickle.load(open('data/cocoRetreival/pool_info.pkl', 'rb'))
        for key, value in image_info.items():
            setattr(self, key, value)

        self.train_imgId2pool = {imageId:i for i, imageId in enumerate(self.train_image_list)}
        self.val_imgId2pool = {imageId:i for i, imageId in enumerate(self.val_image_list)}

        # cache file path data/cache/train_ques
        cap_cache_path = "data/cocoRetreival/cache/" + name + "_cap.pkl"
        if not os.path.exists(cap_cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cap_cache_path, 'wb'))
        else:
            print('loading entries from %s' %(cap_cache_path))
            self._entries = cPickle.load(open(cap_cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self._entries:
            sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            tokens = [
                self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                for w in sentence_tokens
            ]
            tokens = tokens[:self._max_caption_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_caption_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_caption_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_caption_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self._entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids


    def __getitem__(self, index):
        entry = self._entries[index]
        image_id = entry["image_id"]

        features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        image_mask = [1] * (int(num_boxes))

        while len(image_mask) < 37:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()

        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]


        if random.random() > 0.5:
            rand_img_id_pool = self.train_hard_pool[self.train_imgId2pool[image_id]]
            rand_pool_i = int(np.random.randint(len(rand_img_id_pool)))
            pool_img_id = int(rand_img_id_pool[rand_pool_i])
            img_id = self.train_image_list[pool_img_id]

            if random.random() > 0.5:
                rand_features, rand_num_boxes, rand_boxes, _ = self._image_features_reader[img_id]

                rand_image_mask = [1] * (int(rand_num_boxes))
                while len(rand_image_mask) < 37:
                    rand_image_mask.append(0)
                rand_features = torch.tensor(rand_features).float()
                rand_image_mask = torch.tensor(rand_image_mask).long()
                rand_spatials = torch.tensor(rand_boxes).float()

                rand_caption = caption
                rand_input_mask = input_mask
                rand_segment_ids = segment_ids
            else:
                rand_entry = self._entries[random.choice(self.imgid2entry[img_id])]
                rand_caption = rand_entry["token"]
                rand_input_mask = rand_entry["input_mask"]
                rand_segment_ids = rand_entry["segment_ids"]

                rand_features = feature
                rand_image_mask = image_mask
                rand_spatials = spatials         
        else:
            # same feature, grab a random caption.
            rand_entry = self._entries[np.random.randint(len(self._entries)-1)]
            rand_caption = rand_entry["token"]
            rand_input_mask = rand_entry["input_mask"]
            rand_segment_ids = rand_entry["segment_ids"]
            rand_features = feature
            rand_image_mask = image_mask
            rand_spatials = spatials      

        features = torch.stack((features, rand_features), dim=0)
        spatials = torch.stack((spatials, rand_spatials), dim=0)
        image_mask = torch.stack((image_mask, rand_image_mask), dim=0)
        caption = torch.stack((caption, rand_caption), dim=0)
        input_mask = torch.stack((input_mask, rand_input_mask), dim=0)
        segment_ids = torch.stack((segment_ids, rand_segment_ids), dim=0)

        return features, spatials, image_mask, caption, input_mask, segment_ids

    def __len__(self):
        return len(self._entries)
