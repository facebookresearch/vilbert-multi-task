import os
import json
import _pickle as cPickle

# import cPickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer

from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader


os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)

def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """
    if name == 'train' or name == 'val':
        question_path = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name)
        questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        answers = cPickle.load(open(answer_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

    elif name  == 'trainval':
        question_path_train = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % 'train')
        questions_train = sorted(json.load(open(question_path_train))["questions"], key=lambda x: x["question_id"])
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % 'train')
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % 'val')
        questions_val = sorted(json.load(open(question_path_val))["questions"], key=lambda x: x["question_id"])
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % 'val')
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])

        questions = questions_train + questions_val[:-1000]
        answers = answers_train + answers_val[:-1000]

    elif name == 'minval':
        question_path_val = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % 'val')
        questions_val = sorted(json.load(open(question_path_val))["questions"], key=lambda x: x["question_id"])
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % 'val')
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])        

        questions = questions_val[-1000:]
        answers = answers_val[-1000:]

    elif name == 'test':
        question_path_test = os.path.join(dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % 'test')
        questions_test = sorted(json.load(open(question_path_test))["questions"], key=lambda x: x["question_id"])
        answer_path_test = os.path.join(dataroot, "cache", "%s_target.pkl" % 'test')
        answers_test = cPickle.load(open(answer_path_test, "rb"))
        answers_test = sorted(answers_test, key=lambda x: x["question_id"])        

        questions = questions_test
        answers = answers_test    

    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question["question_id"], answer["question_id"])
        assert_eq(question["image_id"], answer["image_id"])
        entries.append(_create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        name: str,
        image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        dataroot="data",
        padding_index: int = 0,
    ):
        super().__init__()
        # assert name in ["train", "val"]

        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_ans_candidates = len(self.ans2label)

        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index

        self.entries = _load_dataset(dataroot, name)

        # cache file path data/cache/train_ques
        # ques_cache_path = "data/VQA/cache/" + name + "_ques.pkl"
        # if not os.path.exists(ques_cache_path):
        self.tokenize()
        self.tensorize()
            # cPickle.dump(self.entries, open(ques_cache_path, 'wb'))
        # else:
            # self.entries = cPickle.load(open(ques_cache_path, "rb"))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.tokenize(entry["question"])
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

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
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        image_mask = [1] * (int(num_boxes))
        while len(image_mask) < 37:
            image_mask.append(0)

        features = torch.tensor(features).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(boxes).float()

        question = entry["q_token"]
        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, image_mask, question, target, input_mask, segment_ids, int(question_id)

    def __len__(self):
        return len(self.entries)
