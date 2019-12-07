# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Dict, List
import random
import os

import torch
from torch.utils.data import Dataset
import numpy as np
import _pickle as cPickle
import json_lines

from pytorch_transformers.tokenization_bert import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader
import pdb
import csv
import sys


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _converId(img_id):

    img_id = img_id.split("-")
    if "train" in img_id[0]:
        new_id = int(img_id[1])
    elif "val" in img_id[0]:
        new_id = int(img_id[1])
    elif "test" in img_id[0]:
        new_id = int(img_id[1])
    else:
        pdb.set_trace()

    return new_id


def _load_annotationsQ_A(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            # det_names = metadata_fn["names"]
            det_names = ""
            question = annotation["question"]
            if split == "test":
                ans_label = 0
            else:
                ans_label = annotation["answer_label"]

            img_id = _converId(annotation["img_id"])
            img_fn = annotation["img_fn"]
            anno_id = int(annotation["annot_id"].split("-")[1])
            entries.append(
                {
                    "question": question,
                    "img_fn": img_fn,
                    "answers": annotation["answer_choices"],
                    "metadata_fn": annotation["metadata_fn"],
                    "target": ans_label,
                    "img_id": img_id,
                    "anno_id": anno_id,
                }
            )

    return entries


def _load_annotationsQA_R(annotations_jsonpath, split):
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""
    entries = []
    with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
        for annotation in json_lines.reader(f):
            # metadata_fn = json.load(open(os.path.join('data/VCR/vcr1images', annotation["metadata_fn"]), 'r'))
            if split == "test":
                # for each answer
                for answer in annotation["answer_choices"]:
                    question = annotation["question"] + ["[SEP]"] + answer
                    img_id = _converId(annotation["img_id"])
                    ans_label = 0
                    img_fn = annotation["img_fn"]
                    anno_id = int(annotation["annot_id"].split("-")[1])
                    entries.append(
                        {
                            "question": question,
                            "img_fn": img_fn,
                            "answers": annotation["rationale_choices"],
                            "metadata_fn": annotation["metadata_fn"],
                            "target": ans_label,
                            "img_id": img_id,
                        }
                    )
            else:
                det_names = ""
                question = (
                    annotation["question"]
                    + ["[SEP]"]
                    + annotation["answer_choices"][annotation["answer_label"]]
                )
                ans_label = annotation["rationale_label"]
                # img_fn = annotation["img_fn"]
                img_id = _converId(annotation["img_id"])
                img_fn = annotation["img_fn"]

                anno_id = int(annotation["annot_id"].split("-")[1])
                entries.append(
                    {
                        "question": question,
                        "img_fn": img_fn,
                        "answers": annotation["rationale_choices"],
                        "metadata_fn": annotation["metadata_fn"],
                        "target": ans_label,
                        "img_id": img_id,
                        "anno_id": anno_id,
                    }
                )

    return entries


class VCRDataset(Dataset):
    def __init__(
        self,
        task: str,
        dataroot: str,
        annotations_jsonpath: str,
        split: str,
        image_features_reader: ImageFeaturesH5Reader,
        gt_image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        bert_model,
        padding_index: int = 0,
        max_seq_length: int = 40,
        max_region_num: int = 60,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        if task == "VCR_Q-A":
            self._entries = _load_annotationsQ_A(annotations_jsonpath, split)
        elif task == "VCR_QA-R":
            self._entries = _load_annotationsQA_R(annotations_jsonpath, split)
        else:
            assert False
        self._split = split
        self._image_features_reader = image_features_reader
        self._gt_image_features_reader = gt_image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_seq_length
        self._max_region_num = max_region_num
        self._bert_model = bert_model
        self.num_labels = 1
        self.dataroot = dataroot

        self._names = []
        with open(os.path.join(dataroot, "unisex_names_table.csv")) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if row[1] != "name":
                    self._names.append(row[1])

        # cache file path data/cache/train_ques
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
                + "_"
                + str(max_region_num)
                + "_vcr_fn.pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task
                + "_"
                + split
                + "_"
                + str(max_seq_length)
                + "_"
                + str(max_region_num)
                + "_vcr_fn.pkl",
            )

        if not os.path.exists(cache_path):
            self.tokenize()
            self.tensorize()
            cPickle.dump(self._entries, open(cache_path, "wb"))
        else:
            self._entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        count = 0
        for entry in self._entries:
            metadata_fn = json.load(
                open(
                    os.path.join(self.dataroot, "vcr1images", entry["metadata_fn"]), "r"
                )
            )
            det_names = metadata_fn["names"]
            random_names = self.generate_random_name(det_names)
            # replace with name
            tokens_a, mask_a = self.replace_det_with_name(
                entry["question"], random_names
            )
            tokens_a = self._tokenizer.encode(" ".join(tokens_a))

            input_ids_all = []
            # co_attention_mask_all = []
            input_mask_all = []
            segment_ids_all = []

            for answer in entry["answers"]:
                tokens_b, mask_b = self.replace_det_with_name(answer, random_names)

                # self._truncate_seq_pair(
                #     tokens_a, tokens_b, mask_a, mask_b, self._max_caption_length - 3
                # )
                tokens_b = self._tokenizer.encode(" ".join(tokens_b))
                self._truncate_seq_pair(
                    tokens_b, self._max_caption_length - 3 - len(tokens_a)
                )

                if "roberta" in self._bert_model:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 2)
                else:
                    segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                input_ids = self._tokenizer.add_special_tokens_sentences_pair(
                    tokens_a, tokens_b
                )

                input_mask = [1] * len(input_ids)
                # Zero-pad up to the sequence length.
                while len(input_ids) < self._max_caption_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    # co_attention_mask.append(-1)

                assert len(input_ids) == self._max_caption_length
                assert len(input_mask) == self._max_caption_length
                assert len(segment_ids) == self._max_caption_length

                # co_attention_mask_all.append(co_attention_mask)
                input_ids_all.append(input_ids)
                input_mask_all.append(input_mask)
                segment_ids_all.append(segment_ids)

            # entry["co_attention_mask"] = co_attention_mask_all
            entry["input_ids"] = input_ids_all
            entry["input_mask"] = input_mask_all
            entry["segment_ids"] = segment_ids_all

            count += 1

    def tensorize(self):

        for entry in self._entries:
            input_ids = torch.from_numpy(np.array(entry["input_ids"]))
            entry["input_ids"] = input_ids

            input_mask = torch.from_numpy(np.array(entry["input_mask"]))
            entry["input_mask"] = input_mask

            segment_ids = torch.from_numpy(np.array(entry["segment_ids"]))
            entry["segment_ids"] = segment_ids

    def generate_random_name(self, det_names):
        random_name = []
        for name in det_names:
            if name == "person":
                word = random.choice(self._names)
            else:
                word = name
            random_name.append(word)

        return random_name

    def replace_det_with_name(self, inputs, random_names):
        tokens = []
        mask = []
        for w in inputs:
            if isinstance(w, str):
                word = w
                det = -1
                word_token = self._tokenizer.tokenize(word)
                mask += [det] * len(word_token)
                tokens += word_token
            else:
                for idx in w:
                    word = random_names[idx]
                    word_token = self._tokenizer.tokenize(word)
                    mask += [idx] * len(word_token)
                    tokens += word_token

        return tokens, mask

    # def _truncate_seq_pair(self, tokens_a, tokens_b, mask_a, mask_b, max_length):
    #     """Truncates a sequence pair in place to the maximum length."""

    #     # This is a simple heuristic which will always truncate the longer sequence
    #     # one token at a time. This makes more sense than truncating an equal percent
    #     # of tokens from each, since if one sequence is very short then each token
    #     # that's truncated likely contains more information than a longer sequence.
    #     while True:
    #         total_length = len(tokens_a) + len(tokens_b)
    #         if total_length <= max_length:
    #             break
    #         if len(tokens_a) > len(tokens_b):
    #             tokens_a.pop()
    #             mask_a.pop()
    #         else:
    #             tokens_b.pop()
    #             mask_b.pop()

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def __getitem__(self, index):

        entry = self._entries[index]

        image_id = entry["img_id"]
        img_query = entry["metadata_fn"][:-5] + ".jpg"
        features, num_boxes, boxes, _ = self._image_features_reader[img_query]

        boxes = boxes[:num_boxes]
        features = features[:num_boxes]

        gt_features, gt_num_boxes, gt_boxes, _ = self._gt_image_features_reader[
            img_query
        ]

        # merge two features.
        features[0] = (features[0] * num_boxes + gt_features[0] * gt_num_boxes) / (
            num_boxes + gt_num_boxes
        )

        # merge two boxes, and assign the labels.
        gt_boxes = gt_boxes[1:gt_num_boxes]
        gt_features = gt_features[1:gt_num_boxes]
        gt_num_boxes = gt_num_boxes - 1

        gt_box_preserve = min(self._max_region_num - 1, gt_num_boxes)
        gt_boxes = gt_boxes[:gt_box_preserve]
        gt_features = gt_features[:gt_box_preserve]
        gt_num_boxes = gt_box_preserve

        num_box_preserve = min(self._max_region_num - int(gt_num_boxes), int(num_boxes))
        boxes = boxes[:num_box_preserve]
        features = features[:num_box_preserve]

        # concatenate the boxes
        mix_boxes = np.concatenate((boxes, gt_boxes), axis=0)
        mix_features = np.concatenate((features, gt_features), axis=0)
        mix_num_boxes = num_box_preserve + int(gt_num_boxes)

        image_mask = [1] * (mix_num_boxes)
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        mix_boxes_pad[:mix_num_boxes] = mix_boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = mix_features[:mix_num_boxes]

        # appending the target feature.
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        input_ids = entry["input_ids"]
        input_mask = entry["input_mask"]
        segment_ids = entry["segment_ids"]
        target = int(entry["target"])

        if self._split == "test":
            # anno_id = entry["anno_id"]
            anno_id = 0  # entry["anno_id"]
        else:
            anno_id = entry["img_id"]

        # co_attention_idxs = entry["co_attention_mask"]
        co_attention_mask = torch.zeros(
            (len(entry["input_ids"]), self._max_region_num, self._max_caption_length)
        )

        # for ii, co_attention_idx in enumerate(co_attention_idxs):
        #     for jj, idx in enumerate(co_attention_idx):
        #         if idx != -1 and idx + num_box_preserve < self._max_region_num:
        #             co_attention_mask[ii, idx + num_box_preserve, jj] = 1

        return (
            features,
            spatials,
            image_mask,
            input_ids,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            anno_id,
        )

    def __len__(self):
        return len(self._entries)
