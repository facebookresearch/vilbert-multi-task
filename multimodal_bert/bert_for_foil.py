from torch import nn
from multimodal_bert.bert import BertConfig, BertModel, BertPreTrainedModel


class MultiModalBertForFoilClassification(BertPreTrainedModel):
    """Multi-modal BERT for the FOIL classification task. This task is trivially similar to binary
    VQA, where the caption is treated as a "question" and the goal is to answer whether the image
    and caption match or not.

    This module is composed of the Multi-modal BERT model with a linear layer on top of the pooled
    outputs from visual and textual BERT.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    >>> # Already been converted into WordPiece token ids
    >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    >>> input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    >>> token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    >>> config = BertConfig(
    ...     vocab_size_or_config_json_file=32000,
    ...     hidden_size=768,
    ...     num_hidden_layers=12,
    ...     num_attention_heads=12,
    ...     intermediate_size=3072
    ... )

    >>> model = MultiModalBertForFoilClassification(config, num_labels=2)
    >>> logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config: BertConfig, num_labels: int = 2):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_txt,
        input_imgs,
        image_loc,
        token_type_ids=None,
        attention_mask=None,
        output_all_encoded_layers=True,
        image_attention_mask=None,
    ):
        pass
