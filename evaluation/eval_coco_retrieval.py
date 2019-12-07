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
import sys

import numpy as np
from time import gmtime, strftime
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from multimodal_bert.datasets import COCORetreivalDatasetTrain, COCORetreivalDatasetVal
from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader


import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features_h5path",
        default="/coc/pskynet2/jlu347/multi-modal-bert/data/flick30k/flickr30k.h5",
    )

    # Required parameters
    parser.add_argument(
        "--val_file",
        default="data/flick30k/all_data_final_test_set0_2014.jsonline",
        type=str,
        help="The input train corpus.",
    )

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
        default="result",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )

    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        # required=True,
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
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=50,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.01,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
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
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--from_pretrained",
        action="store_true",
        help="Wheter the tensor is from pretrained.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Wheter to use the baseline model (single bert).",
    )

    parser.add_argument(
        "--zero_shot", action="store_true", help="Wheter directly evaluate."
    )

    args = parser.parse_args()

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from multimodal_bert.bert import MultiModalBertForImageCaptionRetrieval
        from multimodal_bert.bert import BertForMultiModalPreTraining
    else:
        from multimodal_bert.multi_modal_bert import (
            MultiModalBertForImageCaptionRetrieval,
            BertConfig,
        )
        from multimodal_bert.multi_modal_bert import BertForMultiModalPreTraining

    print(args)
    if args.save_name is not "":
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))

    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    # save all the hidden parameters.
    with open(os.path.join(savePath, "command.txt"), "w") as f:
        print(args, file=f)  # Python 3.x
        print("\n", file=f)
        print(config, file=f)

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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train_examples = None
    num_train_optimization_steps = None

    print("Loading Train Dataset", args.val_file)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    image_features_reader = ImageFeaturesH5Reader(args.features_h5path, True)
    eval_dset = COCORetreivalDatasetVal(args.val_file, image_features_reader, tokenizer)

    config.fast_mode = True
    if args.from_pretrained:
        if args.zero_shot:
            model = BertForMultiModalPreTraining.from_pretrained(
                args.pretrained_weight, config
            )
        else:
            model = MultiModalBertForImageCaptionRetrieval.from_pretrained(
                args.pretrained_weight, config, dropout_prob=0.1
            )
    else:
        if args.zero_shot:
            model = BertForMultiModalPreTraining.from_pretrained(
                args.bert_model, config
            )
        else:
            model = MultiModalBertForImageCaptionRetrieval.from_pretrained(
                args.bert_model, config, dropout_prob=0.1
            )

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
        model = torch.nn.DataParallel(model)

    model.cuda()
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(eval_dset))
    logger.info("  Batch size = %d", args.train_batch_size)

    eval_dataloader = DataLoader(
        eval_dset,
        shuffle=False,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0

    r1, r5, r10, medr, meanr = evaluate(args, model, eval_dataloader)
    print("finish evaluation, save result to %s")

    val_name = args.val_file.split("/")[-1]
    with open(os.path.join(savePath, val_name + "_result.txt"), "w") as f:
        print(
            "r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr),
            file=f,
        )


def evaluate(args, model, dataloader):
    score = 0
    total_loss = 0
    num_data = 0
    count = 0

    score_matrix = np.zeros((5000, 1000))
    target_matrix = np.zeros((5000, 1000))
    rank_matrix = np.ones((5000)) * 1000
    model.eval()
    for batch in tqdm(iter(dataloader)):
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, caption, input_mask, segment_ids, target, caption_idx, image_idx = (
            batch
        )

        features = features.squeeze(0)
        spatials = spatials.squeeze(0)
        image_mask = image_mask.squeeze(0)

        with torch.no_grad():
            if args.zero_shot:
                _, _, logit, _ = model(
                    caption, features, spatials, segment_ids, input_mask, image_mask
                )
                score_matrix[caption_idx, image_idx * 500 : (image_idx + 1) * 500] = (
                    torch.softmax(logit, dim=1)[:, 0].view(-1).cpu().numpy()
                )
                target_matrix[caption_idx, image_idx * 500 : (image_idx + 1) * 500] = (
                    target.float().cpu().numpy()
                )
            else:
                logit = model(
                    caption, features, spatials, segment_ids, input_mask, image_mask
                )
                score_matrix[caption_idx, image_idx * 500 : (image_idx + 1) * 500] = (
                    logit.view(-1).cpu().numpy()
                )
                target_matrix[caption_idx, image_idx * 500 : (image_idx + 1) * 500] = (
                    target.float().cpu().numpy()
                )

            if image_idx.item() == 1:
                rank = np.where(
                    (
                        np.argsort(-score_matrix[caption_idx])
                        == np.where(target_matrix[caption_idx] == 1)[0][0]
                    )
                    == 1
                )[0][0]
                rank_matrix[caption_idx] = rank

                rank_matrix_tmp = rank_matrix[: caption_idx + 1]
                r1 = 100.0 * np.sum(rank_matrix_tmp < 1) / len(rank_matrix_tmp)
                r5 = 100.0 * np.sum(rank_matrix_tmp < 5) / len(rank_matrix_tmp)
                r10 = 100.0 * np.sum(rank_matrix_tmp < 10) / len(rank_matrix_tmp)

                medr = np.floor(np.median(rank_matrix_tmp) + 1)
                meanr = np.mean(rank_matrix_tmp) + 1
                logger.info(
                    "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                    % (count, r1, r5, r10, medr, meanr)
                )

        count += 1

    r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
    r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
    r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

    medr = np.floor(np.median(rank_matrix) + 1)
    meanr = np.mean(rank_matrix) + 1
    logger.info(
        "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
        % (r1, r5, r10, medr, meanr)
    )
    return r1, r5, r10, medr, meanr


if __name__ == "__main__":

    main()
