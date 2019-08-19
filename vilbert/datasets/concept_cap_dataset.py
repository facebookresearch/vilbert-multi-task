import copy
import json
import logging
import math
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def deserialize_lmdb(ds):
    return msgpack.loads(
        ds[1],
        raw=False,
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN,
    )


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat=None, image_target=None, caption=None, is_next=None, lm_labels=None, image_loc=None, num_boxes=None
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes

class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        is_next=None,
        lm_label_ids=None,
        image_feat=None,
        image_target=None,
        image_loc=None,
        image_label=None,
        image_mask=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask

class ConceptCapLoaderTrain(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        predict_feature=False,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        local_rank=-1,
        objective=0,
        visualization=False,
    ):
        TRAIN_DATASET_SIZE = 3119449

        if dist.is_available() and local_rank != -1:

            num_replicas = dist.get_world_size()
            rank = dist.get_rank()

            lmdb_file = os.path.join(corpus_path, "training_feat_part_" + str(rank) + ".lmdb")

            # num_workers = (num_workers // num_replicas) + 1
            # keys = range(TRAIN_DATASET_SIZE)
            # partition_size = math.ceil(TRAIN_DATASET_SIZE / num_replicas)
            # keys = [
            #     keys[i : i + partition_size]
            #     for i in range(0, len(keys), partition_size)
            # ][rank]
            # keys = ["{:0>8d}".format(i).encode() for i in keys]

            # df = td.LMDBData(lmdb_file, shuffle=True, keys=keys)
            # df._size = len(keys)
            # ds = td.MapData(df, deserialize_lmdb)
            # self.num_dataset = len(keys)
        else:
            lmdb_file = os.path.join(corpus_path, "training_feat_all.lmdb")
            print("Loading from %s" % lmdb_file)


        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        ds = td.LocallyShuffleData(ds, cache)
        caption_path = os.path.join(corpus_path, "caption_train.json")
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            36,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            objective=objective,
        )

        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        # self.ds = td.PrefetchData(ds, 1)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        # self.ds = ds
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask, image_id = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask)

            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()

class ConceptCapLoaderVal(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        predict_feature=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    ):
    
        lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")
        caption_path = os.path.join(corpus_path, "caption_val.json")
        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            caption_path,
            tokenizer,
            seq_len,
            36,
            self.num_dataset,
            encoding="utf-8",
            predict_feature=predict_feature,
            visualization=visualization,
            objective=objective,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):
        for batch in self.ds.get_data():
            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            image_loc, image_target, image_label, image_mask, image_id = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            # batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
            # image_loc, image_target, image_label, image_mask, image_id)
            batch = (input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, \
                image_loc, image_target, image_label, image_mask)

            yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return self.ds.size()

class BertPreprocessBatch(object):
    def __init__(
        self,
        caption_path,
        tokenizer,
        seq_len,
        region_len, 
        data_size,
        split="Train",
        encoding="utf-8",
        predict_feature=False,
        visualization=False,
        objective=0,
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, 'r')).values())
        self.visualization = visualization
        self.objective = objective
        
    def __call__(self, data):

        image_feature_wp, image_target_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption = data
        
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes,:4] = image_location_wp

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)

        if self.predict_feature:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)            

        caption, label = self.random_cap(caption)

        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)
        
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            image_id,
        )
        return cur_tensors

    def random_cap(self, caption):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        if self.visualization:
            return caption, 0
        
        if self.objective != 2 and random.random() > 0.5:
            caption = self.get_random_caption()
            label = 1
        else:
            label = 0

        return caption, label

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        # add the hard negative mining objective here.
        rand_doc_idx = random.randint(0, self.num_caps - 1)
        caption = self.captions[rand_doc_idx]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        is_next = example.is_next

        self._truncate_seq_pair(caption, max_seq_length - 2)

        caption, caption_label = self.random_word(caption, tokenizer, is_next)
        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes, is_next)

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
        lm_label_ids = [-1] + caption_label + [-1]
        # image_label = ([-1] + image_label)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)
        # for i in range(36):
        #     # tokens.append(0)
        #     segment_ids.append(0)

        # tokens.append("[SEP]")
        # segment_ids.append(0)
        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        assert len(image_label) == max_region_length

        # if example.guid < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("LM label: %s " % (lm_label_ids))
        #     logger.info("Is next sentence label: %s " % (example.is_next))

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask = np.array(image_mask)
        )
        return features

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

    def random_word(self, tokens, tokenizer, is_next):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            
            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # not sample mask
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes, is_next):
        """
        """
        output_label = []

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability
            
            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label


class ConceptCapLoaderRetrieval(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the 
            GPU, which is faster).
    """

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        predict_feature=False,
        batch_size=512,
        shuffle=False,
        num_workers=10,
        cache=50000,
        drop_last=False,
        cuda=False,
    ):
    
        lmdb_file = "/coc/dataset/conceptual_caption/validation_feat_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/validation_feat_all.lmdb"
        caption_path = "/coc/pskynet2/jlu347/multi-modal-bert/data/conceptual_caption/caption_val.json"

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessRetrieval(
            caption_path,
            tokenizer,
            seq_len,
            36,
            1000,
            encoding="utf-8",
            predict_feature=predict_feature,
        )

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, 1)
        self.ds.reset_state()

        self.batch_size = 1
        self.num_workers = num_workers
        self._entry = []

        self.features_all = np.zeros((1000, 37, 2048), dtype=np.float32)
        self.spatials_all = np.zeros((1000, 37, 5), dtype=np.float32)
        self.image_mask_all = np.zeros((1000, 37), dtype=np.float32)
        self.image_ids = []
        # load first 1000 file here.
        for i, batch in enumerate(self.ds.get_data()):
            if i >= 1000:
                break
            input_ids, input_mask, segment_ids, is_next, image_feat, \
            image_loc, image_mask, image_id, caption = batch

            batch_size = input_ids.shape[0]
            g_image_feat = np.sum(image_feat, axis=1) / np.sum(image_mask, axis=1, keepdims=True)
            image_feat = np.concatenate([np.expand_dims(g_image_feat, axis=1), image_feat], axis=1)
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(np.array([[0,0,1,1,1]], dtype=np.float32), batch_size, axis=0)
            image_loc = np.concatenate([np.expand_dims(g_image_loc, axis=1), image_loc], axis=1)
            
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (input_ids, input_mask, segment_ids, image_id, caption)
            self._entry.append(batch)
            
            self.features_all[i] = image_feat
            self.image_mask_all[i] = np.array(image_mask)
            self.spatials_all[i] = image_loc
            self.image_ids.append(image_id)
            sys.stdout.write('%d/%d\r' % (i, 1000))
            sys.stdout.flush()            


    def __iter__(self):

        for index in range(self.__len__()):
            caption_idx = int(index / 2)
            image_idx = index % 2

            if image_idx == 0:
                image_entries = self.image_ids[:500]
                features_all = self.features_all[:500]
                spatials_all = self.spatials_all[:500]
                image_mask_all = self.image_mask_all[:500]

            else:
                image_entries = self.image_ids[500:]
                features_all = self.features_all[500:]
                spatials_all = self.spatials_all[500:]
                image_mask_all = self.image_mask_all[500:]

            caption, input_mask, segment_ids, txt_image_id, caption = self._entry[caption_idx]
            target_all = np.zeros((500))
            for i, image_id in enumerate(image_entries):
                if image_id == txt_image_id:
                    target_all[i] = 1

            batch = (features_all, spatials_all, image_mask_all, caption, input_mask, segment_ids, target_all, caption_idx, image_idx)
            batch = [torch.tensor(data) for data in batch]
            batch.append(txt_image_id)
            batch.append(caption)
            
            yield batch


    def __len__(self):
        return len(self._entry) * 2


class BertPreprocessRetrieval(object):
    def __init__(
        self,
        caption_path,
        tokenizer,
        seq_len,
        region_len, 
        data_size,
        split="Train",
        encoding="utf-8",
        predict_feature=False,
    ):

        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.num_caps = data_size
        self.captions = list(json.load(open(caption_path, 'r')).values())[:data_size]

    def __call__(self, data):

        image_feature_wp, image_target_wp, image_location_wp, num_boxes,  image_h, image_w, image_id, caption = data
        
        image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes,:4] = image_location_wp

        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)

        label = 0
        
        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
            num_boxes=num_boxes
        )

        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer, self.region_len)
        
        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_mask,
            float(image_id),
            caption,
        )
        return cur_tensors


    def convert_example_to_features(self, example, max_seq_length, tokenizer, max_region_length):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
        IDs, LM labels, input_mask, CLS and SEP tokens etc.
        :param example: InputExample, containing sentence input as strings and is_next label
        :param max_seq_length: int, maximum length of sequence.
        :param tokenizer: Tokenizer
        :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
        """
        image_feat = example.image_feat
        caption = example.caption
        image_loc = example.image_loc
        # image_target = example.image_target
        num_boxes = int(example.num_boxes)
        self._truncate_seq_pair(caption, max_seq_length - 2)
        # caption, caption_label = self.random_word(caption, tokenizer)
        caption_label = None
        # image_feat, image_loc, image_label = self.random_region(image_feat, image_loc, num_boxes)
        image_label = None

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(image_mask) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            is_next=np.array(example.is_next),
            image_feat=image_feat,
            image_loc=image_loc,
            image_mask = np.array(image_mask),
        )
        return features

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
