from __future__ import print_function
import os
import json
import six 
import _pickle as cPickle
#import cPickle
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
import pdb
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)

def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        assert_eq(question['question_id'], answer['question_id'])
        assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class BertDictionary(object):
    def __init__(self, args):
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    @property
    def ntoken(self):
        return len(self.tokenizer.vocab)

    @property
    def padding_idx(self):
        return 0

    def tokenize(self, sentence, add_word):
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

        tokens =[]
        for w in sentence_tokens:
            if w in self.tokenizer.vocab:
                tokens.append(self.tokenizer.vocab[w])
            else:
                tokens.append(self.tokenizer.vocab["[UNK]"])

        return tokens

    def __len__(self):
        return len(self.tokenizer.vocab)

class VQAClassificationDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data'):
        super(VQAClassificationDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name), 'rb'))

        print('loading features from h5 file')
        h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        # with h5py.File(h5_path, 'r') as hf:
        #     self.features = np.array(hf.get('image_features'))
        #     self.spatials = np.array(hf.get('spatial_features'))

        h5file = h5py.File(h5_path, 'r')
        self.features = h5file['image_features']
        self.spatials = h5file['spatial_features']

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)


        # cache file path data/cache/train_ques
        ques_cache_path = 'data/cache/' + name + '_ques.pkl'
        if not os.path.exists(ques_cache_path):
            self.tokenize()
            self.tensorize()
            # cPickle.dump(self.entries, open(ques_cache_path, 'wb'))
        else:
            self.entries = cPickle.load(open(ques_cache_path, 'rb'))

        self.v_dim = self.features.shape[2]
        self.s_dim = self.spatials.shape[2]

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens
            entry['q_input_mask'] = input_mask
            entry['q_segment_ids'] = segment_ids
            
    def tensorize(self):
        # self.features = torch.from_numpy(self.features)
        # self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            q_input_mask = torch.from_numpy(np.array(entry['q_input_mask']))
            entry['q_input_mask'] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry['q_segment_ids']))
            entry['q_segment_ids'] = q_segment_ids

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        features = torch.from_numpy(self.features[entry['image']])
        spatials = torch.from_numpy(self.spatials[entry['image']])

        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        input_mask = entry['q_input_mask']
        segment_ids = entry['q_segment_ids']

        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, spatials, question, target, input_mask, segment_ids

    def __len__(self):
        return len(self.entries)
