import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_annotations(annotations_jsonpath: str) -> Dict[int, List[Dict[str, Any]]]:
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""

    annotations_json: Dict[str, Any] = json.load(open(annotations_jsonpath))

    # Build an index which maps image id with a list of caption annotations.
    entries = []

    for annotation in annotations_json["annotations"]:
        entries.append(
            {"caption": annotation["caption"].lower(), "foil": annotation["foil"], 'image_id':annotation["image_id"]}
        )
    return entries


class FoilClassificationDataset(Dataset):
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
        
        self._entries = _load_annotations(annotations_jsonpath)
        

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_caption_length

        # cache file path data/cache/train_ques
        # foil_cache_path = "data/foil/cache/" + name + "_foil.pkl"
        # if not os.path.exists(foil_cache_path):
        self.tokenize()
        self.tensorize()
            # cPickle.dump(self._entries, open(foil_cache_path, 'wb'))
        # else:
            # self._entries = cPickle.load(open(foil_cache_path, "rb"))

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
        target = int(entry["foil"])
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        return features, spatials, image_mask, caption, target, input_mask, segment_ids

    def __len__(self):
        return len(self._entries)
