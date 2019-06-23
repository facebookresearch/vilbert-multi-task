import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import _pickle as cPickle

import sys
sys.path.append("tools/refer")
from refer import REFER

def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

class ReferExpressionDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str, 
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_seq_length: int = 20,
        max_region_num: int = 60
    ):
        self.split = split
        self.refer = REFER(dataroot, dataset=task,  splitBy='unc')
        self.ref_ids = self.refer.getRefIds(split=split)
        print('%s refs are in split [%s].' % (len(self.ref_ids), split))

        self.num_labels = 1
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self.entries = self._load_annotations()

        self.max_region_num = max_region_num

        cache_path = os.path.join(dataroot, "cache", task + '_' + split + '_' + str(max_seq_length)+ "_" + str(max_region_num) + '.pkl')
        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, 'wb'))
        else:
            print('loading entries from %s' %(cache_path))
            self.entries = cPickle.load(open(cache_path, "rb"))

    def _load_annotations(self):

        # annotations_json: Dict[str, Any] = json.load(open(annotations_jsonpath))

        # Build an index which maps image id with a list of caption annotations.
        entries = []

        for ref_id in self.ref_ids:
            ref = self.refer.Refs[ref_id]
            image_id = ref['image_id']
            ref_id = ref['ref_id']
            refBox = self.refer.getRefBox(ref_id)
            for sent, sent_id in zip(ref['sentences'], ref['sent_ids']):
                caption = sent['raw']
                entries.append(
                    {"caption": caption, 'sent_id':sent_id, 'image_id':image_id, \
                    "refBox": refBox, 'ref_id': ref_id}
                    )
        
        return entries

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for entry in self.entries:
            
            sentence_tokens = self._tokenizer.tokenize(entry["caption"])
            sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

            tokens = [
                self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                for w in sentence_tokens
            ]

            tokens = tokens[:self._max_seq_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), self._max_seq_length)
            entry["token"] = tokens
            entry["input_mask"] = input_mask
            entry["segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            token = torch.from_numpy(np.array(entry["token"]))
            entry["token"] = token

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids


    def __getitem__(self, index):
        entry = self.entries[index]

        image_id = entry["image_id"]
        ref_box = entry["refBox"]

        ref_box = [ref_box[0], ref_box[1], ref_box[0]+ref_box[2], ref_box[1]+ref_box[3]]
        features, num_boxes, boxes, boxes_ori = self._image_features_reader[image_id]

        boxes_ori = boxes_ori[:num_boxes]
        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        if self.split == 'train':
            gt_features, gt_num_boxes, gt_boxes, gt_boxes_ori = self._gt_image_features_reader[image_id]

            # merge two boxes, and assign the labels. 
            gt_boxes_ori = gt_boxes_ori[1:gt_num_boxes]
            gt_boxes = gt_boxes[1:gt_num_boxes]
            gt_features = gt_features[1:gt_num_boxes]

            # concatenate the boxes
            mix_boxes_ori = np.concatenate((boxes_ori, gt_boxes_ori), axis=0)
            mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
            mix_features = np.concatenate((features, gt_features), axis=0)
            mix_num_boxes = min(int(num_boxes + int(gt_num_boxes) - 1), self.max_region_num)
            # given the mix boxes, and ref_box, calculate the overlap. 
            mix_target = iou(torch.tensor(mix_boxes_ori[:,:4]).float(), torch.tensor([ref_box]).float())
            mix_target[mix_target<0.5] = 0

        else:
            mix_boxes_ori = boxes_ori
            mix_boxes = boxes
            mix_features = features
            mix_num_boxes = min(int(num_boxes), self.max_region_num)
            mix_target = iou(torch.tensor(mix_boxes_ori[:,:4]).float(), torch.tensor([ref_box]).float())

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self.max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self.max_region_num, 5))
        mix_features_pad = np.zeros((self.max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes
        mix_features_pad[:mix_num_boxes] = mix_features

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        target = torch.zeros((self.max_region_num,1)).float()
        target[:mix_num_boxes] = mix_target

        spatials_ori = torch.tensor(mix_boxes_ori).float()
        co_attention_mask = torch.zeros((self.max_region_num, self._max_seq_length))

        caption = entry["token"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]

        return features, spatials, image_mask, caption, target, input_mask, segment_ids, co_attention_mask, image_id

    def __len__(self):
        return len(self.entries)
