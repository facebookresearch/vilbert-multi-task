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
import pdb

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

from multimodal_bert.datasets import VQAClassificationDataset
from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader
from parallel.data_parallel import DataParallel

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    # Data files for VQA task.
    parser.add_argument("--features_h5path", default="data/coco/test2015.h5")
    parser.add_argument(
        "--train_file",
        default="data/VQA/training",
        type=str,
        # required=True,
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
        default="save",
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
        "--use_location", action="store_true", help="whether use location."
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
        default=30,
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
        default=20,
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
    parser.add_argument("--split", default="test", type=str, help="train or trainval.")

    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )
    args = parser.parse_args()

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from multimodal_bert.bert import MultiModalBertForVQA
    else:
        from multimodal_bert.multi_modal_bert import MultiModalBertForVQA, BertConfig

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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # train_examples = None
    num_train_optimization_steps = None

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    image_features_reader = ImageFeaturesH5Reader(args.features_h5path, True)

    if args.split == "minval":
        eval_dset = VQAClassificationDataset(
            "minval", image_features_reader, tokenizer, dataroot="data/VQA"
        )
    elif args.split == "test":
        eval_dset = VQAClassificationDataset(
            "test", image_features_reader, tokenizer, dataroot="data/VQA"
        )
    elif args.split == "val":
        eval_dset = VQAClassificationDataset(
            "val", image_features_reader, tokenizer, dataroot="data/VQA"
        )
    elif args.split == "test-dev":
        eval_dset = VQAClassificationDataset(
            "test-dev", image_features_reader, tokenizer, dataroot="data/VQA"
        )

    num_labels = eval_dset.num_ans_candidates
    if args.from_pretrained:
        model = MultiModalBertForVQA.from_pretrained(
            args.pretrained_weight, config, num_labels=num_labels
        )
    else:
        model = MultiModalBertForVQA.from_pretrained(
            args.bert_model, config, num_labels=num_labels
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
        model = DataParallel(model, use_chuncks=args.use_chunk)

    model.cuda()

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dset))
    logger.info("  Batch size = %d", args.train_batch_size)

    eval_dataloader = DataLoader(
        eval_dset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()

    model.train(False)
    eval_score, bound = evaluate(args, model, eval_dataloader)
    logger.info("\teval score: %.2f (%.2f)" % (100 * eval_score, 100 * bound))


class TBlogger:
    def __init__(self, log_dir):
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)


def evaluate(args, model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    results = []
    for batch in tqdm(iter(dataloader)):
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_id = (
            batch
        )
        with torch.no_grad():
            pred = model(
                question, features, spatials, segment_ids, input_mask, image_mask
            )
            logits = torch.max(pred, 1)[1].data  # argmax

            for i in range(logits.size(0)):
                results.append(
                    {
                        "question_id": question_id[i].item(),
                        "answer": dataloader.dataset.label2ans[logits[i].item()],
                    }
                )

        # batch_score = compute_score_with_logits(pred, target.cuda()).sum()
        # score += batch_score.item()
        # upper_bound += (target.max(1)[0]).sum()
        # num_data += pred.size(0)

    json.dump(results, open(args.save_name + ".json", "w"))

    # score = score / len(dataloader.dataset)
    # upper_bound = upper_bound / len(dataloader.dataset)
    # return score, upper_bound


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def lr_lambda_update(i_iter):
    warmup_iterations = 1000
    warmup_factor = 0.2
    lr_ratio = 0.1
    lr_steps = [15000, 18000, 20000, 21000]
    if i_iter <= warmup_iterations:
        alpha = float(i_iter) / float(warmup_iterations)
        return warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect([], i_iter)
        return pow(lr_ratio, idx)


if __name__ == "__main__":

    main()
