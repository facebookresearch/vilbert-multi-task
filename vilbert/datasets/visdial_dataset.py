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
import csv
import sys

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_annotations(annotations_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []

    print('Loading dataset from %s' %annotations_jsonpath)
    annotations = json.load(open(annotations_jsonpath, 'r'))['data']
    print('Finish loading ...')
    for dialog in annotations['dialogs']:
        image_id = dialog['image_id']
        caption = dialog['caption']
        for i, qa in enumerate(dialog['dialog']):
            if i == 0:
                fact = caption
            else:
                fact = [dialog['dialog'][i-1]['question'], dialog['dialog'][i-1]['answer']]

            entries.append({'image_id':image_id, 'question':qa['question'], 'fact':fact, \
                    'round':i, 'answer_options':qa['answer_options'], 'gt_index':qa['gt_index']})

    return entries, annotations['questions'], annotations['answers']

class VisDialDataset(Dataset):
    def __init__(
        self,
        name: str,
        annotations_jsonpath: str,
        image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_caption_length: int = 40,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_caption_length
        self._max_region_num = 37

        self._entries, questions, answers = _load_annotations(annotations_jsonpath)
        self._questions, self._answers = self.tokenizeQA(questions, answers)

        self.CLS = self._tokenizer.convert_tokens_to_ids(["[CLS]"])
        self.SEP = self._tokenizer.convert_tokens_to_ids(["[SEP]"])

    def tokenizeQA(self, questions, answers):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        question_token = []
        answer_token = []

        for question in questions:
            # replace with name
            question_token.append(self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(question)))

        for answer in answers:
            # replace with name
            answer_token.append(self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(answer)))

        return question_token, answer_token

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def __getitem__(self, index):
        
        pdb.set_trace()        
        entry = self._entries[index]

        image_id = entry["img_id"]
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        # boxes = boxes[:num_boxes]
        # features = features[:num_boxes]
        
        # image_mask = [1] * (num_boxes)
        # while len(image_mask) < self._max_region_num:
            # image_mask.append(0)

        # boxes_pad = np.zeros((self._max_region_num, 5))
        # features_pad = np.zeros((self._max_region_num, 2048))

        # boxes_pad[:num_boxes] = mix_boxes[:num_boxes]
        # features_pad[:num_boxes] = mix_features[:num_boxes]

        # appending the target feature.
        # features = torch.tensor(features_pad).float()
        # image_mask = torch.tensor(image_mask).long()
        # spatials = torch.tensor(boxes_pad).float()


        ques = self._questions[entry['question']]
        if entry['round'] == 0:
            fact = self.CLS + self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(entry['fact'])) + self.SEP
        else:
            fact = self._questions[entry['fact'][0]] + self.SEP + self._answers[entry['fact'][1]] + self.SEP 
        tokens_a = fact + ques
        # sample random answer pairs.

        answer_candidate = []
        answer_candidate.append(entry['gt_index'])
        for i in range(5):
            while True:
                randIdx = np.random.randint(len(entry['answer_options']))
                if randIdx != entry['gt_index']:
                    break
            answer_candidate.append(randIdx)

        random.shuffle(answer_candidate)
        
        for ans_idx in answer_candidate:
            tokens_b = self._answers[entry['answer_options'][ans_idx]]
            self._truncate_seq_pair(tokens_a)
            input_ids = tokens_a

        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        target = int(entry["target"])

        co_attention_idxs = entry["co_attention_mask"]
        co_attention_mask = torch.zeros((4, self._max_region_num, self._max_caption_length))

        for ii, co_attention_idx in enumerate(co_attention_idxs):
            for jj, idx in enumerate(co_attention_idx):
                if idx != -1:
                    co_attention_mask[ii, idx, ii] = 1

        return features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask

    def __len__(self):
        return len(self._entries)
