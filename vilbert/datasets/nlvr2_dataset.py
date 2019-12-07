# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import jsonlines
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(item):
    entry = {
        "question_id": item["question_id"],
        "image_id_0": item["image_id_0"],
        "image_id_1": item["image_id_1"],
        "sentence": item["sentence"],
        "answer": item,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'dev', 'test'
    """
    if name == "train" or name == "dev" or name == "test":
        annotations_path = os.path.join(dataroot, "%s.json" % name)
        with jsonlines.open(annotations_path) as reader:

            # Build an index which maps image id with a list of hypothesis annotations.
            items = []
            count = 0
            for annotation in reader:
                # logger.info(annotation)
                dictionary = {}
                dictionary["id"] = annotation["identifier"]
                dictionary["image_id_0"] = (
                    "-".join(annotation["identifier"].split("-")[:-1]) + "-img0"
                )
                dictionary["image_id_1"] = (
                    "-".join(annotation["identifier"].split("-")[:-1]) + "-img1"
                )
                dictionary["question_id"] = count
                dictionary["sentence"] = str(annotation["sentence"])
                dictionary["labels"] = [0 if str(annotation["label"]) == "False" else 1]
                dictionary["scores"] = [1.0]
                items.append(dictionary)
                count += 1
    else:
        assert False, "data split is not recognized."

    entries = []
    for item in items:
        entries.append(_create_entry(item))
    return entries


class NLVR2Dataset(Dataset):
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
        max_region_num=37,
    ):
        super().__init__()
        self.split = split
        self.num_labels = 2
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
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
                + ".pkl",
            )
        else:
            cache_path = os.path.join(
                dataroot,
                "cache",
                task + "_" + split + "_" + str(max_seq_length) + ".pkl",
            )

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split)
            self.tokenize(max_seq_length)
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            # tokens = self._tokenizer.tokenize(entry["sentence"])
            # tokens = ["[CLS]"] + tokens + ["[SEP]"]

            # tokens = [
            #     self._tokenizer.vocab.get(w, self._tokenizer.vocab["[UNK]"])
            #     for w in tokens
            # ]

            tokens = self._tokenizer.encode(entry["sentence"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            answer = entry["answer"]
            labels = np.array(answer["labels"])
            scores = np.array(answer["scores"], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry["answer"]["labels"] = labels
                entry["answer"]["scores"] = scores
            else:
                entry["answer"]["labels"] = None
                entry["answer"]["scores"] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id_0 = entry["image_id_0"]
        image_id_1 = entry["image_id_1"]
        question_id = entry["question_id"]
        features_0, num_boxes_0, boxes_0, _ = self._image_features_reader[image_id_0]
        features_1, num_boxes_1, boxes_1, _ = self._image_features_reader[image_id_1]

        mix_num_boxes = min(
            int(num_boxes_0) + int(num_boxes_1), self._max_region_num * 2
        )
        mix_boxes_pad = np.zeros((self._max_region_num * 2, 5))
        mix_features_pad = np.zeros((self._max_region_num * 2, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num * 2:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = np.concatenate((boxes_0, boxes_1), axis=0)[
            :mix_num_boxes
        ]

        mix_features_pad[:mix_num_boxes] = np.concatenate(
            (features_0, features_1), axis=0
        )[:mix_num_boxes]

        img_segment_ids = np.zeros((mix_features_pad.shape[0]))
        img_segment_ids[: boxes_0.shape[0]] = 0
        img_segment_ids[boxes_0.shape[0] :] = 1

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        img_segment_ids = torch.tensor(img_segment_ids).long()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros(
            (self._max_region_num * 2, self._max_seq_length)
        )
        target = torch.zeros(self.num_labels)

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        if labels is not None:
            target.scatter_(0, labels, scores)

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
