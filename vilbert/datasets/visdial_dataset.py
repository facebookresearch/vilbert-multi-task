# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
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


def _load_dataset(annotations_jsonpath, clean_datasets):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    captions = []
    remove_ids = []
    if clean_datasets:
        remove_ids = np.load(os.path.join(dataroot, "cache", "genome_test_ids.npy"))
        remove_ids = [int(x) for x in remove_ids]
    print("Loading dataset from %s" % annotations_jsonpath)
    annotations = json.load(open(annotations_jsonpath, "r"))["data"]
    print("Finish loading ...")
    for i, dialog in enumerate(annotations["dialogs"]):
        image_id = dialog["image_id"]
        if int(image_id) in remove_ids:
            continue
        captions.append(dialog["caption"])
        entries.append({"image_id": image_id, "dialog": dialog["dialog"], "caption": i})

    return entries, annotations["questions"], annotations["answers"], captions


class VisDialDataset(Dataset):
    def __init__(
        self,
        task,
        dataroot,
        annotations_jsonpath,
        split,
        image_features_reader,
        gt_image_features_reader,
        tokenizer,
        bert_model,
        clean_datasets,
        padding_index=0,
        max_seq_length=16,
        max_region_num=101,
    ):

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_seq_length = max_seq_length
        self._max_region_num = max_region_num
        self._total_seq_length = 50
        self.num_labels = 1

        self.max_round_num = 3
        self.max_num_option = 4
        self.ans_option = 100
        self.CLS = self._tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        self.SEP = self._tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

        clean_train = "_cleaned" if clean_datasets else ""

        if "roberta" in bert_model:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + "roberta"
                + "_"
                + str(max_seq_length)
                + clean_train
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
            )

        if not os.path.exists(cache_path):
            self._entries, questions, answers, captions = _load_dataset(
                annotations_jsonpath, clean_datasets
            )
            self._questions, self._answers, self._captions = self.tokenizeQA(
                questions, answers, captions
            )
            file_save = {}
            file_save["entries"] = self._entries
            file_save["questions"] = self._questions
            file_save["answers"] = self._answers
            file_save["captions"] = self._captions
            cPickle.dump(file_save, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            file_save = cPickle.load(open(cache_path, "rb"))
            self._entries = file_save["entries"]
            self._questions = file_save["questions"]
            self._answers = file_save["answers"]
            self._captions = file_save["captions"]

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
            question_token.append(
                self._tokenizer.convert_tokens_to_ids(
                    self._tokenizer.tokenize(question)
                )
            )

        for answer in answers:
            # replace with name
            answer_token.append(
                self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(answer))
            )

        for caption in captions:
            # replace with name
            caption_token.append(
                self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(caption))
            )

        return question_token, answer_token, caption_token

    def _truncate_seq(self, tokens_a, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break

            tokens_a.pop(0)

        return tokens_a

    def __getitem__(self, index):

        entry = self._entries[index]
        image_id = entry["image_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()

        # Let's sample one dialog at a time.
        caption = self._captions[entry["caption"]]

        input_ids_all = []
        input_mask_all = []
        segment_ids_all = []

        for rnd in range(10):
            ques = self._questions[entry["dialog"][rnd]["question"]]
            # fact is all previous question+answer
            tokens_fact = []
            for j in range(rnd):
                if rnd - self.max_round_num <= j:
                    fact_q = self._questions[entry["dialog"][j]["question"]]
                    fact_a = self._answers[entry["dialog"][j]["answer"]]
                    if len(tokens_fact) == 0:
                        tokens_fact = tokens_fact + fact_q + [self.SEP] + fact_a
                    else:
                        tokens_fact = (
                            tokens_fact + [self.SEP] + fact_q + [self.SEP] + fact_a
                        )

            token_q = ques

            if len(tokens_fact) == 0:
                tokens_f = caption
            else:
                tokens_f = tokens_fact + [self.SEP] + caption
            answer_candidate = []
            answer_candidate.append(entry["dialog"][rnd]["gt_index"])
            rand_idx = np.random.permutation(self.ans_option)
            count = 0
            while len(answer_candidate) < self.max_num_option:
                if rand_idx[count] != entry["dialog"][rnd]["gt_index"]:
                    answer_candidate.append(rand_idx[count])
                count += 1

            input_ids_rnd = []
            input_mask_rnd = []
            segment_ids_rnd = []

            for i, ans_idx in enumerate(answer_candidate):
                tokens_a = self._answers[
                    entry["dialog"][rnd]["answer_options"][ans_idx]
                ]
                tokens_f_new = self._truncate_seq(
                    copy.deepcopy(tokens_f),
                    self._total_seq_length - len(token_q) - len(tokens_a) - 4,
                )

                tokens = []
                segment_ids = []

                tokens.append(self.CLS)
                segment_ids.append(0)
                for token in token_q:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(self.SEP)
                segment_ids.append(0)

                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(1)

                tokens.append(self.SEP)
                segment_ids.append(1)

                for token in tokens_f_new:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(self.SEP)
                segment_ids.append(0)

                input_mask = [1] * (len(tokens))
                # Zero-pad up to the sequence length.
                while len(tokens) < self._total_seq_length:
                    tokens.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)

                input_ids_rnd.append(tokens)
                input_mask_rnd.append(input_mask)
                segment_ids_rnd.append(segment_ids)

            input_ids_all.append(input_ids_rnd)
            input_mask_all.append(input_mask_rnd)
            segment_ids_all.append(segment_ids_rnd)

        input_ids = torch.from_numpy(np.array(input_ids_all))
        input_mask = torch.from_numpy(np.array(input_mask_all))
        segment_ids = torch.from_numpy(np.array(segment_ids_all))
        co_attention_mask = torch.zeros(
            (10, self.max_num_option, self._max_region_num, self._total_seq_length)
        )
        target = torch.zeros(10).long()
        return (
            features,
            spatials,
            image_mask,
            input_ids,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            image_id,
        )

    def __len__(self):
        return len(self._entries)
