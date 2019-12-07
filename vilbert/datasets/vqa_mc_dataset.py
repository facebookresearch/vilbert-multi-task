# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_transformers.tokenization_bert import BertTokenizer
import random
from ._image_features_reader import ImageFeaturesH5Reader
import sys
import pdb

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, option, answer):
    answer.pop("image_id")
    answer.pop("question_id")

    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "option": option["answer"][:4],
        "answer": answer["multiple_choice_answer"],
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """

    options_path = "VQA_bert_base_4layer_4conect-pretrained_finetune"

    if name == "train" or name == "val":
        question_path = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name
        )
        questions = sorted(
            json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
        )

        answer_path = os.path.join(dataroot, "v2_mscoco_%s2014_annotations.json" % name)
        answers = sorted(
            json.load(open(question_path))["annotations"],
            key=lambda x: x["question_id"],
        )

        option_path = os.path.join("results", options_path, "%s_others.json" % name)
        options = sorted(json.load(open(option_path)), key=lambda x: x["question_id"])

    elif name == "trainval":
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )

        answer_path_train = os.path.join(
            dataroot, "v2_mscoco_%s2014_annotations.json" % "train"
        )
        answers_train = sorted(
            json.load(open(answer_path_train))["annotations"],
            key=lambda x: x["question_id"],
        )

        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(
            dataroot, "v2_mscoco_%s2014_annotations.json" % "val"
        )
        answers_val = sorted(
            json.load(open(answer_path_val))["annotations"],
            key=lambda x: x["question_id"],
        )

        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

        option_path_train = os.path.join(
            "results", options_path, "%s_others.json" % "train"
        )
        options_train = sorted(
            json.load(open(option_path_train)), key=lambda x: x["question_id"]
        )
        option_path_val = os.path.join(
            "results", options_path, "%s_others.json" % "val"
        )
        options_val = sorted(
            json.load(open(option_path_val)), key=lambda x: x["question_id"]
        )
        options = options_train + options_val[:-3000]

    elif name == "minval":
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(
            dataroot, "v2_mscoco_%s2014_annotations.json" % "val"
        )
        answers_val = sorted(
            json.load(open(answer_path_val))["annotations"],
            key=lambda x: x["question_id"],
        )
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]

        option_path = os.path.join("results", options_path, "%s_others.json" % "val")
        options = sorted(json.load(open(option_path)), key=lambda x: x["question_id"])
        options = options[-3000:]

    elif name == "test":
        question_path_test = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test"
        )
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test

        option_path = os.path.join("results", options_path, "%s_others.json" % "test")
        options = sorted(json.load(open(option_path)), key=lambda x: x["question_id"])

    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question, option in zip(questions, options):
            assert_eq(question["question_id"], option["question_id"])
            entries.append(_create_entry(question, option, None))
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        for question, answer, option in zip(questions, answers, options):
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            assert_eq(question["question_id"], option["question_id"])
            entries.append(_create_entry(question, option, answer))
    return entries


class VQAMultipleChoiceDataset(Dataset):
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
        super().__init__()
        self.split = split
        self.num_labels = 1
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        cache_path = os.path.join(
            dataroot, "cache", task + "_" + split + "_" + str(max_seq_length) + ".pkl"
        )
        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

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

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """

        count = 0
        for entry in self.entries:
            option = entry["option"]

            if self.split != "test":
                # replace one answer if it is not exist in option
                ans_exist = False
                if entry["answer"] in option:
                    ans_exist = True

                if not ans_exist:
                    random.shuffle(option)
                    option.pop()
                    option.append(entry["answer"])

                # identify the target.
                for i, ans in enumerate(option):
                    if ans == entry["answer"]:
                        target = i

            tokens_all = []
            input_mask_all = []
            segment_ids_all = []
            for i, ans in enumerate(option):

                tokens_a = self._tokenizer.tokenize(entry["question"])
                tokens_b = self._tokenizer.tokenize(ans)
                tokens_a, tokens_b = self._truncate_seq_pair(
                    tokens_a, tokens_b, max_length - 3
                )

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

                tokens = [
                    self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
                    for w in tokens
                ]

                tokens = tokens[:max_length]
                segment_ids = [0] * len(tokens)
                input_mask = [1] * len(tokens)

                if len(tokens) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self._padding_index] * (max_length - len(tokens))
                    tokens = tokens + padding
                    input_mask += padding
                    segment_ids += padding

                assert_eq(len(tokens), max_length)
                tokens_all.append(tokens)
                input_mask_all.append(input_mask)
                segment_ids_all.append(segment_ids)

            entry["q_token"] = tokens_all
            entry["q_input_mask"] = input_mask_all
            entry["q_segment_ids"] = segment_ids_all
            if self.split != "test":
                entry["target"] = target

            sys.stdout.write("%d/%d\r" % (count, len(self.entries)))
            sys.stdout.flush()
            count += 1

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((4, self._max_region_num, self._max_seq_length))

        if "test" not in self.split:
            target = entry["target"]

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
        )

    def __len__(self):
        return len(self.entries)
