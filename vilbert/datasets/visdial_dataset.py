import json
from typing import Any, Dict, List
import random
import os
import logging

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
import csv
import sys
import copy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _load_dataset(annotations_jsonpath):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    captions = []
    print('Loading dataset from %s' %annotations_jsonpath)
    annotations = json.load(open(annotations_jsonpath, 'r'))['data']
    print('Finish loading ...')
    for i, dialog in enumerate(annotations['dialogs']):
        image_id = dialog['image_id']
        captions.append(dialog['caption'])
        for j, qa in enumerate(dialog['dialog']):
            if j == 0:
                previous_qa = [None, None]
            else:
                previous_qa = [dialog['dialog'][j-1]['question'], dialog['dialog'][j-1]['answer']]

            entries.append({'image_id':image_id, 'question':qa['question'], 'caption': i, 'previous_qa':previous_qa, \
                    'round':i, 'answer_options':qa['answer_options'], 'gt_index':qa['gt_index']})

    return entries, annotations['questions'], annotations['answers'], captions

class VisDialDataset(Dataset):
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
        max_seq_length: int = 16,
        max_region_num: int = 37,
    ):
        self.split = split
        self.num_labels = 1
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        self.CLS = self._tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.SEP = self._tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

        cache_path = os.path.join('data', task, "cache", task + '_' + split + '_' + str(max_seq_length)+'.pkl')
        if not os.path.exists(cache_path):
            self._entries, questions, answers, captions = _load_dataset(annotations_jsonpath)
            self._questions, self._answers, self._captions = self.tokenizeQA(questions, answers, captions)
            file_save = {}
            file_save['entries'] = self._entries
            file_save['questions'] = self._questions
            file_save['answers'] = self._answers
            file_save['captions'] = self._captions
            cPickle.dump(file_save, open(cache_path, 'wb'))
        else:
            logger.info("Loading from %s" %cache_path)
            file_save = cPickle.load(open(cache_path, "rb"))
            self._entries = file_save['entries'] 
            self._questions = file_save['questions'] 
            self._answers = file_save['answers'] 
            self._captions = file_save['captions'] 

    def tokenizeQA(self, questions, answers, captions):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        question_token = []
        answer_token = []
        caption_token = []

        for question in questions:
            # replace with name
            question_token.append(self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(question)))

        for answer in answers:
            # replace with name
            answer_token.append(self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(answer)))

        for caption in captions:
            # replace with name
            caption_token.append(self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(caption)))

        return question_token, answer_token, caption_token

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

        return tokens_a, tokens_b
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

        ques = self._questions[entry['question']]
        if entry['previous_qa'][0] != None:
            fact = self._questions[entry['previous_qa'][0]] + self._answers[entry['previous_qa'][1]]
        else:
            fact = []

        caption = self._captions[entry['caption']]

        tokens_q = caption + [self.CLS] + fact + [self.CLS] + ques
        # sample random answer pairs.

        answer_candidate = []
        answer_candidate.append(entry['gt_index'])
        for i in range(3):
            while True:
                randIdx = np.random.randint(len(entry['answer_options']))
                if randIdx != entry['gt_index']:
                    break
            answer_candidate.append(randIdx)
        
        random.shuffle(answer_candidate)
        
        input_ids_all = []
        input_mask_all = []
        segment_ids_all = []

        for i, ans_idx in enumerate(answer_candidate):
            if ans_idx == entry['gt_index']:
                target = i

            tokens_b = self._answers[entry['answer_options'][ans_idx]]
            tokens_a, tokens_b = self._truncate_seq_pair(copy.deepcopy(tokens_q), tokens_b, self._max_seq_length-3)
            tokens = []
            segment_ids = []

            tokens.append(self.CLS)
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            
            tokens.append(self.SEP)
            segment_ids.append(0)

            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)

            tokens.append(self.SEP)
            segment_ids.append(1)
            
            input_mask = [1] * (len(tokens))
            # Zero-pad up to the sequence length.
            while len(tokens) < self._max_seq_length:
                tokens.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            input_ids_all.append(tokens)
            input_mask_all.append(input_mask)
            segment_ids_all.append(segment_ids)

        input_ids = torch.from_numpy(np.array(input_ids_all))
        input_mask = torch.from_numpy(np.array(input_mask_all))
        segment_ids = torch.from_numpy(np.array(segment_ids_all))
        co_attention_mask = torch.zeros((4, self._max_region_num, self._max_seq_length))

        return features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask, image_id

    def __len__(self):
        return len(self._entries)
