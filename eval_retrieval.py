# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict
import sys
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn

from vilbert.task_utils import (
    LoadDatasetEval,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
    EvaluatingModel,
)
from vilbert.vilbert import VILBertForVLTasks, BertForMultiModalPreTraining
from vilbert.basebert import BaseBertForVLTasks

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_config.json",
        type=str,
        help="The config file which specified the model details.",
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
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--use_chunk",
        default=0,
        type=float,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--tasks", default="", type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--zero_shot", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument("--split", default="", type=str, help="which split to use.")
    parser.add_argument("--batch_size", default=1, type=int, help="which split to use.")
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        help="whether to use task specific tokens for the multi-task learning.",
    )
    args = parser.parse_args()
    with open("vilbert_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
    else:
        from vilbert.vilbert import BertConfig

    task_names = []
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)

    # timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0]
    if "/" in args.from_pretrained:
        timeStamp = args.from_pretrained.split("/")[1]
    else:
        timeStamp = args.from_pretrained

    savePath = os.path.join(args.output_dir, timeStamp)

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(
        open("config/" + args.bert_model + "_weight_name.json", "r")
    )

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

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu and not os.path.exists(savePath):
        os.makedirs(savePath)

    task_batch_size, task_num_iters, task_ids, task_datasets_val, task_dataloader_val = LoadDatasetEval(
        args, task_cfg, args.tasks.split("-")
    )

    num_labels = max([dataset.num_labels for dataset in task_datasets_val.values()])

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    config.fast_mode = True
    if args.zero_shot:
        model = BertForMultiModalPreTraining.from_pretrained(
            args.from_pretrained, config
        )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, deay_allreduce=True)

    elif n_gpu > 1:
        model = nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    print("***** Running training *****")
    print("  Num Iters: ", task_num_iters)
    print("  Batch size: ", task_batch_size)

    model.eval()
    # when run evaluate, we run each task sequentially.
    for task_id in task_ids:
        results = []
        others = []

        score_matrix = np.zeros((5000, 1000))
        target_matrix = np.zeros((5000, 1000))
        rank_matrix = np.ones((5000)) * 1000
        count = 0

        for i, batch in enumerate(task_dataloader_val[task_id]):
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
            features, spatials, image_mask, question, input_mask, segment_ids, target, caption_idx, image_idx = (
                batch
            )
            task_tokens = (
                question.new().resize_(question.size(0), 1).fill_(int(task_id[4:]))
            )

            if task_id in ["TASK7", "TASK8"]:
                batch_size = features.size(0)
                features = features.squeeze(0)
                spatials = spatials.squeeze(0)
                image_mask = image_mask.squeeze(0)

            with torch.no_grad():
                if args.zero_shot:
                    _, _, vil_logit, _ = model(
                        question,
                        features,
                        spatials,
                        segment_ids,
                        input_mask,
                        image_mask,
                        task_ids=task_tokens,
                    )

                    score_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (torch.softmax(vil_logit, dim=1)[:, 0].view(-1).cpu().numpy())
                    target_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (target.view(-1).float().cpu().numpy())

                else:
                    _, _, vil_logit, _, _, _, _, _, _ = model(
                        question,
                        features,
                        spatials,
                        segment_ids,
                        input_mask,
                        image_mask,
                        task_ids=task_tokens,
                    )
                    score_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (vil_logit.view(-1).cpu().numpy())
                    target_matrix[
                        caption_idx, image_idx * 500 : (image_idx + 1) * 500
                    ] = (target.view(-1).float().cpu().numpy())

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
                    print(
                        "%d Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
                        % (count, r1, r5, r10, medr, meanr)
                    )

                    results.append(np.argsort(-score_matrix[caption_idx]).tolist()[:20])
            count += 1

        r1 = 100.0 * np.sum(rank_matrix < 1) / len(rank_matrix)
        r5 = 100.0 * np.sum(rank_matrix < 5) / len(rank_matrix)
        r10 = 100.0 * np.sum(rank_matrix < 10) / len(rank_matrix)

        medr = np.floor(np.median(rank_matrix) + 1)
        meanr = np.mean(rank_matrix) + 1

        print("************************************************")
        print(
            "Final r1:%.3f, r5:%.3f, r10:%.3f, mder:%.3f, meanr:%.3f"
            % (r1, r5, r10, medr, meanr)
        )
        print("************************************************")

        if args.split:
            json_path = os.path.join(savePath, args.split)
        else:
            json_path = os.path.join(savePath, task_cfg[task_id]["val_split"])
        json.dump(results, open(json_path + "_result.json", "w"))
        json.dump(others, open(json_path + "_others.json", "w"))


if __name__ == "__main__":

    main()
