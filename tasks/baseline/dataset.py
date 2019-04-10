import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import glob, os
import h5py
import pdb 
import json
import random
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class CaptionDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.num_file = 1

        self.image_files = {}
        for i in range(self.num_file):
            imagePath = os.path.join(corpus_path, str(i)+'.h5')
            imageReader = h5py.File(imagePath, 'r', libver='latest', swmr=True)
            self.image_files[i] = imageReader

        self.caption_files = {}
        for i in range(self.num_file):
            captionPath = os.path.join(corpus_path, str(i)+'.json')
            self.caption_files[i] = json.load(open(captionPath, 'r'))
        
        # count all the image-caption pairs.
        self.num_caps = sum([len(i) for i in self.caption_files.values()])
        print('total image captions in the dataset %d' %(self.num_caps))

        self.index_map = []
        # given an index, also create the map that give the file and associate index. 
        count = 0
        for i, caption_file in self.caption_files.items():
            for j, caption in enumerate(caption_file):
                self.index_map.append([i,j]) 

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.num_caps

    def __getitem__(self, item):
        # cur_id = self.sample_counter
        # self.sample_counter += 1

        img_feat, caption, is_next_label, image_target, image_id = self.random_cap(item)

        # tokenize
        tokens_caption = self.tokenizer.tokenize(caption)

        # combine to one sample
        cur_example = InputExample(guid=image_id, image_feat=img_feat, caption=tokens_caption, \
                                                is_next=is_next_label, image_target=image_target)

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.lm_label_ids),
                       torch.tensor(cur_features.is_next),
                       torch.tensor(cur_features.image_feat),
                       torch.tensor(cur_features.image_target))

        return cur_tensors

    def random_cap(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        image_feature, image_target, image_id, caption = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_caption()
            label = 1

        return image_feature, caption, label, image_target, image_id

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        file_id, index_id = self.index_map[item]
        image_feature = self.image_files[file_id]['features'][index_id]
        image_target = self.image_files[file_id]['prediction'][index_id]
        caption = self.caption_files[file_id][index_id]['caption']
        image_id = self.image_files[file_id]['image_ids'][index_id]

        if np.sum(image_target) == 0:
            print(image_id)

        return image_feature, image_target, image_id, caption

    def get_random_caption(self):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        rand_doc_idx = random.randint(0, self.num_caps-1)
        file_id, index_id = self.index_map[rand_doc_idx]
        caption = self.caption_files[file_id][index_id]['caption']

        return caption

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            #keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line

def _truncate_seq_pair(len_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len_a + len(tokens_b)
        if total_length <= max_length:
            break
        
        tokens_b.pop()

class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(self, guid, image_feat, caption=None, is_next=None, image_target=None, lm_labels=None):
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
        self.guid = guid
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence
        self.lm_labels = lm_labels  # masked words for language model
        self.image_target = image_target

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, is_next, lm_label_ids, image_feat, image_target):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_target = image_target

def random_word(tokens, tokenizer):
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
        if prob < 0.15:
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
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_region(image_feat):
    """
    """
    output_label = []

    for i in range(36):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                image_feat[i] = 0

            # 10% randomly change token to random token
            # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            
            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            output_label.append(-2)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return image_feat, output_label    


def convert_example_to_features(example, max_seq_length, tokenizer):
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
    image_target = example.image_target

    _truncate_seq_pair(36, caption, max_seq_length - 3)

    caption, caption_label = random_word(caption, tokenizer)

    image_feat, image_label = random_region(image_feat)
    # concatenate lm labels and account for CLS, SEP, SEP
    # lm_label_ids = ([-1] + caption_label + [-1] + image_label + [-1])
    lm_label_ids = ([-1] + image_label + [-1] + caption_label + [-1])

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
    for i in range(36):
        # tokens.append(0)        
        segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in caption:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_ids = [0] * 36 + input_ids
    input_mask = [1] * (len(input_ids))
    
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

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("LM label: %s " % (lm_label_ids))
        logger.info("Is next sentence label: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             lm_label_ids=lm_label_ids,
                             is_next=example.is_next,
                             image_feat = image_feat,
                             image_target = image_target)
    return features
