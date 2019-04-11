import json
from typing import Any, Dict, List
import random

import torch
from torch.utils.data import Dataset

from pytorch_pretrained_bert.tokenization import BertTokenizer
from ._image_features_reader import ImageFeaturesH5Reader


def _load_annotations(annotations_jsonpath: str) -> Dict[int, List[Dict[str, Any]]]:
    """Build an index out of FOIL annotations, mapping each image ID with its corresponding captions."""

    annotations_json: Dict[str, Any] = json.load(open(annotations_jsonpath))

    # Build an index which maps image id with a list of caption annotations.
    entries: Dict[int, List[Dict[str, Any]]] = {}

    for annotation in annotations_json["annotations"]:
        if annotation["image_id"] not in entries:
            entries[annotation["image_id"]] = []

        # Only keep relevant fields for now: 'caption', 'foil'
        entries[annotation["image_id"]].append(
            {"caption": annotation["caption"], "foil": annotation["foil"]}
        )
    return entries


class FoilClassificationDataset(Dataset):
    def __init__(
        self,
        annotations_jsonpath: str,
        image_features_reader: ImageFeaturesH5Reader,
        tokenizer: BertTokenizer,
        padding_index: int = 0,
        max_caption_length: int = 20,
    ):
        # All the keys in `self._entries` would be present in `self._image_features_reader`
        self._entries = _load_annotations(annotations_jsonpath)
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer

        self._padding_index = padding_index
        self._max_caption_length = max_caption_length

        self.tokenize()

        # Hold a list of Image IDs to index the dataset.
        self._image_ids = list(self._entries.keys())

    def tokenize(self):
        """Tokenizes the captions.

        This will add caption_tokens in each entry of the dataset.
        -1 represents nil, and should be treated as padding_idx in embedding.
        """
        for image_id in self._entries:
            for i in range(len(self._entries[image_id])):
                sentence_tokens = self._tokenizer.tokenize(self._entries[image_id][i]["caption"])
                sentence_tokens = ["[CLS]"] + sentence_tokens + ["[SEP]"]

                tokens = []
                for w in sentence_tokens:
                    if w in self._tokenizer.vocab:
                        tokens.append(self._tokenizer.vocab[w])
                    else:
                        tokens.append(self._tokenizer.vocab["[UNK]"])
                tokens = tokens[: self._max_caption_length]
                if len(tokens) < self._max_caption_length:
                    # Note here we pad in front of the sentence
                    padding = [self._padding_index] * (self._max_caption_length - len(tokens))
                    tokens = padding + tokens
                self._entries[image_id][i]["caption"] = tokens

    def __getitem__(self, index):
        image_id = self._image_ids[index]
        features = torch.tensor(self._image_features_reader[image_id])

        # Pick a random caption.
        entry = random.choice(self._entries[image_id])

        caption = torch.tensor(entry["caption"])
        target = torch.tensor(int(entry["foil"]))

        # TODO (kd): update code to return spatial features
        spatials = -1

        return features, spatials, caption, target

    def __len__(self):
        return len(self._entries)
