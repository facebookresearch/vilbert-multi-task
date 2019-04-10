import copy
import json
import logging
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self, image_feat, image_target, caption=None, is_next=None, lm_labels=None, image_loc=None
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids,
        input_mask,
        segment_ids,
        is_next,
        lm_label_ids,
        image_feat,
        image_target,
        image_loc,
        image_label,
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


class CaptionLoader(object):
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
        use_location=False,
        predict_feature=False,
        hard_negative=False,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=50000,
        drop_last=False,
        cuda=False,
    ):
        # enumerate standard imagenet augmentors
        # imagenet_augmentors = fbresnet_augmentor(mode == 'train')
        # load the lmdb if we can find it
        # lmdb_loc = os.path.join(os.environ['IMAGENET'],'ILSVRC-%s.lmdb'%mode)

        lmdb_file = "/coc/dataset/conceptual_caption/training_feat_all.lmdb"
        if not os.path.exists(lmdb_file):
            lmdb_file = "data/conceptual_caption/training_feat_all.lmdb"

        print("Loading from %s" % lmdb_file)

        ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(
            corpus_path,
            tokenizer,
            seq_len,
            self.num_dataset,
            encoding="utf-8",
            use_location=use_location,
            predict_feature=predict_feature,
            hard_negative=hard_negative,
        )

        ds = td.LocallyShuffleData(ds, cache)
        ds = td.PrefetchData(ds, 5000, 1)
        ds = td.MapData(ds, preprocess_function)
        ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            # input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label = batch
            yield tuple(torch.tensor(data) for data in batch)

    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        data_size,
        encoding="utf-8",
        use_location=False,
        predict_feature=False,
        hard_negative=False,
    ):

        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.predict_feature = predict_feature
        self.use_location = use_location
        self.hard_negative = hard_negative
        self.num_file = 20
        self.num_caps = data_size
        self.caption_files = {}
        for i in range(self.num_file):
            captionPath = os.path.join(corpus_path, str(i) + "_cap.json")
            self.caption_files[i] = json.load(open(captionPath, "r"))

        self.index_map = []
        # given an index, also create the map that give the file and associate index.
        count = 0
        for i, caption_file in self.caption_files.items():
            for j, caption in enumerate(caption_file):
                self.index_map.append([i, j])

    def __call__(self, data):
        image_feature, image_target, image_location, hard_negative_list, file_id, index_id = data
        if self.predict_feature:
            image_target = copy.deepcopy(image_feature)

        caption, label = self.random_cap(file_id, index_id, hard_negative_list)
        tokens_caption = self.tokenizer.tokenize(caption)
        cur_example = InputExample(
            image_feat=image_feature,
            image_target=image_target,
            caption=tokens_caption,
            is_next=label,
            image_loc=image_location,
        )
        # transform sample to features
        cur_features = self.convert_example_to_features(cur_example, self.seq_len, self.tokenizer)
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
        )
        return cur_tensors

    def random_cap(self, file_id, index_id, hard_negative_list):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        if random.random() > 0.5:
            caption = self.caption_files[file_id][index_id]["caption"]
            label = 0
        else:
            caption = self.get_random_caption(file_id, index_id, hard_negative_list)
            label = 1

        return caption, label

    def get_random_caption(self, file_id, index_id, hard_negative_list):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        # add the hard negative mining objective here.

        if self.hard_negative:
            if random.random() < 0.5:
                rand_idx = random.randint(1, 49)
                rand_doc_idx = hard_negative_list[rand_idx]
                caption = self.caption_files[file_id_ori][int(rand_doc_idx)]["caption"]
                # caption_ori = self.caption_files[file_id_ori][index_id_ori]['caption']
            else:
                rand_doc_idx = random.randint(0, self.num_caps - 1)
                file_id, index_id = self.index_map[rand_doc_idx]
                caption = self.caption_files[file_id][index_id]["caption"]
        else:
            rand_doc_idx = random.randint(0, self.num_caps - 1)
            file_id, index_id = self.index_map[rand_doc_idx]
            caption = self.caption_files[file_id][index_id]["caption"]

        return caption

    def convert_example_to_features(self, example, max_seq_length, tokenizer):
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
        self._truncate_seq_pair(caption, max_seq_length - 2)
        caption, caption_label = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label = self.random_region(image_feat, image_loc)

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

    def random_word(self, tokens, tokenizer):
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
                    logger.warning(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                    )
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc):
        """
        """
        output_label = []

        for i in range(36):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                    image_loc[i] = 0
                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(-2)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label


class SchedualSampler(Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self._num_per_file = data_source.num_per_file
        if self._num_samples is not None and replacement is False:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )
        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):

        iterlist = []
        offset = 0
        for n in self._num_per_file:
            iterlist.append(list(torch.randperm(n).numpy() + offset))
            offset += n

        random.shuffle(iterlist)
        # flatten the list
        iterlist = [item for sublist in iterlist for item in sublist]

        # if self.replacement:
        # return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(iterlist)

    def __len__(self):
        return self.num_samples


class CaptionDataset(Dataset):
    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding="utf-8",
        use_location=False,
        predict_feature=False,
        hard_negative=False,
        corpus_lines=None,
        on_memory=True,
    ):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc
        self.predict_feature = predict_feature
        self.hard_negative = hard_negative
        # for loading samples directly from file
        self.use_location = use_location
        self.num_file = 20

        self.envs = {}
        for i in range(self.num_file):
            # if True:
            imagePath = os.path.join(corpus_path, str(i) + "_new.lmdb")
            self.envs[i] = lmdb.open(
                imagePath, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
            )

        if not self.predict_feature:
            print("\nUse the soft label as the visual target...")
            self.target_envs = {}
            for i in range(self.num_file):
                imagePath = os.path.join(corpus_path, str(i) + "_pred.lmdb")
                self.target_envs[i] = lmdb.open(
                    imagePath,
                    max_readers=1,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )

        print("\nloading caption file...", end=" ")
        self.caption_files = {}
        for i in range(self.num_file):
            # if True:
            print(i, end=" ")
            captionPath = os.path.join(corpus_path, str(i) + "_cap.json")
            self.caption_files[i] = json.load(open(captionPath, "r"))

        if self.use_location:
            print("\nloading location file...", end=" ")
            self.location_files = {}
            for i in range(self.num_file):
                print(i, end=" ")
                locationPath = os.path.join(corpus_path, str(i) + "_loc_size.json")
                self.location_files[i] = json.load(open(locationPath, "r"))

        if self.hard_negative:
            print("\nloading hard negative list...", end=" ")
            self.hard_negative = {}
            for i in range(self.num_file):
                print(i, end=" ")
                hardnegativePath = os.path.join(corpus_path, str(i) + "_hard_negative.npy")
                self.hard_negative[i] = np.load(hardnegativePath)

        # count all the image-caption pairs.
        self.num_per_file = [len(i) for i in self.caption_files.values()]
        self.num_caps = sum(self.num_per_file)
        print("\ntotal image captions in the dataset %d" % (self.num_caps))

        self.index_map = []
        # given an index, also create the map that give the file and associate index.
        count = 0
        for i, caption_file in self.caption_files.items():
            for j, caption in enumerate(caption_file):
                self.index_map.append([i, j])

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence". Additionally, we start counting at 0.
        return self.num_caps

    def __getitem__(self, item):

        # cur_id = self.sample_counter
        # self.sample_counter += 1
        img_feat, img_target, caption, is_next_label, image_loc, image_id = self.random_cap(item)

        # tokenize
        tokens_caption = self.tokenizer.tokenize(caption)

        # combine to one sample
        cur_example = InputExample(
            guid=int(image_id),
            image_feat=img_feat,
            image_target=img_target,
            caption=tokens_caption,
            is_next=is_next_label,
            image_loc=image_loc,
        )

        # transform sample to features
        cur_features = convert_example_to_features(cur_example, self.seq_len, self.tokenizer)

        cur_tensors = (
            torch.tensor(cur_features.input_ids),
            torch.tensor(cur_features.input_mask),
            torch.tensor(cur_features.segment_ids),
            torch.tensor(cur_features.lm_label_ids),
            torch.tensor(cur_features.is_next),
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            torch.Tensor(cur_features.image_label),
        )

        return cur_tensors

    def random_cap(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50% these are two subsequent sentences
        from one doc. With 50% the second sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """

        image_feature, image_target, image_location, image_id, caption = self.get_corpus_line(
            index
        )
        if random.random() > 0.5:
            label = 0
        else:
            caption = self.get_random_caption(index)
            label = 1

        return image_feature, image_target, caption, label, image_location, image_id

    def get_corpus_line(self, item):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines from the same doc.
        :param item: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        file_id, index_id = self.index_map[item]
        caption = self.caption_files[file_id][index_id]["caption"]
        image_id = self.caption_files[file_id][index_id]["image_id"]

        env = self.envs[file_id]
        with env.begin(write=False) as txn:
            image_feature = np.frombuffer(txn.get(image_id.encode()), dtype="float32").reshape(
                36, 2048
            )

        image_feature = torch.tensor(image_feature)

        if not self.predict_feature:
            target_env = self.target_envs[file_id]
            with target_env.begin(write=False) as txn:
                image_target = np.frombuffer(txn.get(image_id.encode()), dtype="float32").reshape(
                    36, 1601
                )

            image_target = torch.tensor(image_target)
        else:
            image_target = image_feature.clone()

        if self.use_location:
            image_location = self.location_files[file_id][index_id]["location"]
            width = self.location_files[file_id][index_id]["width"]
            height = self.location_files[file_id][index_id]["height"]
            # ===================================================
            # since the width and height are from the original image, not from the
            # detectron output, this is just a hack.
            im_size_min = min(width, height)
            im_size_max = max(width, height)
            MAX_SIZE = 1333
            SCALE = 800
            im_scale = float(SCALE) / float(im_size_min)
            if np.round(im_scale * im_size_max) > MAX_SIZE:
                im_scale = float(MAX_SIZE) / float(im_size_max)

            # ===================================================
            image_location = torch.tensor(image_location)
            image_location[:, 0] = image_location[:, 0] / width / im_scale
            image_location[:, 2] = image_location[:, 2] / width / im_scale

            image_location[:, 1] = image_location[:, 1] / height / im_scale
            image_location[:, 3] = image_location[:, 3] / height / im_scale
        else:
            image_location = torch.rand(36, 4)

        # if np.sum(image_target) == 0:
        # print(image_id)

        return image_feature, image_target, image_location, image_id, caption

    def get_random_caption(self, index):
        """
        Get random caption from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document we're processing.

        # add the hard negative mining objective here.

        if self.hard_negative:
            if random.random() < 0.5:
                file_id_ori, index_id_ori = self.index_map[index]
                rand_idx = random.randint(1, 49)
                rand_doc_idx = self.hard_negative[file_id_ori][index_id_ori][rand_idx]
                caption = self.caption_files[file_id_ori][int(rand_doc_idx)]["caption"]
                # caption_ori = self.caption_files[file_id_ori][index_id_ori]['caption']
            else:
                rand_doc_idx = random.randint(0, self.num_caps - 1)
                file_id, index_id = self.index_map[rand_doc_idx]
                caption = self.caption_files[file_id][index_id]["caption"]
        else:
            rand_doc_idx = random.randint(0, self.num_caps - 1)
            file_id, index_id = self.index_map[rand_doc_idx]
            caption = self.caption_files[file_id][index_id]["caption"]

        return caption

    def get_next_line(self):
        """ Gets next line of random_file and starts over when reaching end of file"""
        try:
            line = next(self.random_file).strip()
            # keep track of which document we are currently looking at to later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line


def _truncate_seq_pair(tokens_b, max_length):
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
                logger.warning(
                    "Cannot find token '{}' in vocab. Using [UNK] insetad".format(token)
                )
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_region(image_feat, image_loc):
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
                image_feat[i].zero_()
                image_loc[i].zero_()
            # 10% randomly change token to random token
            # elif prob < 0.9:
            # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token
            # append current token to output (we will predict these later)
            output_label.append(-2)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return image_feat, image_loc, output_label


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
    image_loc = example.image_loc
    image_target = example.image_target
    _truncate_seq_pair(caption, max_seq_length - 2)
    caption, caption_label = random_word(caption, tokenizer)

    image_feat, image_loc, image_label = random_region(image_feat, image_loc)

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
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        is_next=example.is_next,
        image_feat=image_feat,
        image_target=image_target,
        image_loc=image_loc,
        image_label=image_label,
    )
    return features