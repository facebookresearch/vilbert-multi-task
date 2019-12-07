# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BERT finetuning runner."""

import argparse
import json
import logging
import os
import random
from io import open
from bisect import bisect

from time import gmtime, strftime
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multimodal_bert.datasets import ReferExpressionDataset
from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader
from torch.nn import CrossEntropyLoss

from parallel.data_parallel import DataParallel


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Data files for FOIL task.
    parser.add_argument(
        "--features_h5path",
        default="/coc/pskynet2/jlu347/multi-modal-bert/data/referExpression",
    )
    parser.add_argument("--instances-jsonpath", default="data/referExpression")
    parser.add_argument("--task", default="refcoco+")

    # Required parameters
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--pretrained_weight",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )

    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=30,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=128,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--from_pretrained",
        action="store_true",
        help="Wheter the tensor is from pretrained.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Wheter to use the baseline model (single bert).",
    )

    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )

    parser.add_argument(
        "--split",
        default="test",
        type=str,
        help="whether use chunck for parallel training.",
    )

    args = parser.parse_args()

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from multimodal_bert.bert import MultiModalBertForReferExpression
    else:
        from multimodal_bert.multi_modal_bert import (
            MultiModalBertForReferExpression,
            BertConfig,
        )

    # Declare path to save checkpoints.
    print(args)
    config = BertConfig.from_json_file(args.config_file)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(args.seed)

    num_train_optimization_steps = None

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    features_h5path = os.path.join(args.features_h5path, args.task + ".h5")
    gt_features_h5path = os.path.join(args.features_h5path, args.task + "_gt.h5")

    image_features_reader = ImageFeaturesH5Reader(features_h5path, True)

    eval_dset = ReferExpressionDataset(
        args.task,
        args.split,
        args.instances_jsonpath,
        image_features_reader,
        None,
        tokenizer,
    )

    # config = BertConfig.from_json_file(args.config_file)

    num_labels = 2
    if args.from_pretrained:
        model = MultiModalBertForReferExpression.from_pretrained(
            args.pretrained_weight, config, dropout_prob=0.2
        )
    else:
        model = MultiModalBertForReferExpression(config, dropout_prob=0.2)

    print("loading model from %s" % (args.pretrained_weight))
    checkpoint = torch.load(args.pretrained_weight)
    model.load_state_dict(checkpoint)

    if args.fp16:
        model.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = DataParallel(model, use_chuncks=args.use_chunk)

    model.cuda()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(eval_dset))
    logger.info("  Batch size = %d", args.train_batch_size)

    eval_dataloader = DataLoader(
        eval_dset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model.eval()
    eval_loss, eval_score = evaluate(args, model, eval_dataloader)
    logger.info("\teval_loss: %.2f, score: %.2f" % (eval_loss, 100 * eval_score))


def evaluate(args, model, dataloader):
    total_loss = 0
    num_data = 0
    score = 0
    loss_fct = CrossEntropyLoss(ignore_index=-1)

    for batch in dataloader:
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, captions, target, input_mask, segment_ids = (
            batch
        )
        with torch.no_grad():
            logits = model(
                captions, features, spatials, segment_ids, input_mask, image_mask
            )
        loss = instance_bce_with_logits(logits.squeeze(2), target.squeeze(2))

        _, select_idx = torch.max(logits, dim=1)
        select_target = target.squeeze(2).gather(1, select_idx.view(-1, 1))
        # count the accuracy.
        score += torch.sum(select_target > 0.5).item()
        total_loss += loss.sum().item()
        num_data += features.size(0)

    return total_loss / len(dataloader), score / num_data


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


if __name__ == "__main__":
    main()
