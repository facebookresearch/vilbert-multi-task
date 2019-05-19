# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from time import gmtime, strftime
from timeit import default_timer as timer
from bisect import bisect

from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from parallel.data_parallel import DataParallel

from multimodal_bert.datasets import VCRDataset
from multimodal_bert.datasets._image_features_reader import ImageFeaturesH5Reader
from torch.nn import CrossEntropyLoss
import pdb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Data files for FOIL task.
    parser.add_argument("--features_h5path", default="/coc/pskynet2/jlu347/multi-modal-bert/data/VCR/VCR_resnet101_faster_rcnn_genome.h5")
    parser.add_argument("--features_gt_h5path", default="/coc/pskynet2/jlu347/multi-modal-bert/data/VCR/VCR_gt_resnet101_faster_rcnn_genome.h5")

    parser.add_argument(
        "--instances-train-jsonpath", default="data/VCR/train.jsonl"
    )
    parser.add_argument(
        "--instances-val-jsonpath", default="data/VCR/val.jsonl"
    )

    parser.add_argument(
        "--task", default="Q-A", type=str, choices=['Q-A', 'QA-R', 'Q-AR'], help="which task to train?"
    )

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
        default=40,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument("--use_location", action="store_true", help="whether use location.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--train_batch_size", default=30, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10,
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
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
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
        "--num_workers", type=int, default=20, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--from_pretrained", action="store_true", help="Wheter the tensor is from pretrained."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Wheter to use the baseline model (single bert)."
    )


    args = parser.parse_args()
    
    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from multimodal_bert.bert import MultiModalBertForVCR
    else:
        from multimodal_bert.multi_modal_bert import MultiModalBertForVCR, BertConfig

    # Declare path to save checkpoints.
    print(args)
    if args.save_name is not '':
        timeStamp = args.save_name
    else:
        timeStamp = strftime("%d-%b-%y-%X-%a", gmtime())
        timeStamp += "_{:0>6d}".format(random.randint(0, 10e6))
    
    savePath = os.path.join(args.output_dir, timeStamp)

    if not os.path.exists(savePath):
        os.makedirs(savePath)
    
    config = BertConfig.from_json_file(args.config_file)
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    if not args.do_train:
        raise ValueError(
            "Training is currently the only implemented execution option. Please set `do_train`."
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = None
    if args.do_train:

        viz = TBlogger("logs", timeStamp)
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )

        image_features_reader = ImageFeaturesH5Reader(args.features_h5path, True, 36)
        image_features_gt_reader = ImageFeaturesH5Reader(args.features_gt_h5path, True, 81)

        train_dset = VCRDataset(
            "train",
            args.task,
            args.instances_train_jsonpath, 
            image_features_reader, 
            image_features_gt_reader, 
            tokenizer,
            max_caption_length=args.max_seq_length,
        )

        eval_dset = VCRDataset(
            "val", 
            args.task,
            args.instances_val_jsonpath, 
            image_features_reader, 
            image_features_gt_reader, 
            tokenizer,
            max_caption_length=args.max_seq_length,

        )

        num_train_optimization_steps = (
            int(len(train_dset) / args.train_batch_size / args.gradient_accumulation_steps)
            * args.num_train_epochs
        )
        if args.local_rank != -1:
            num_train_optimization_steps = (
                num_train_optimization_steps // torch.distributed.get_world_size()
            )

    config = BertConfig.from_json_file(args.config_file)

    num_labels = 2
    if args.from_pretrained:
        model = MultiModalBertForVCR.from_pretrained(
            args.pretrained_weight, config, num_labels=num_labels
        )
    else:
        model = MultiModalBertForVCR(config, num_labels)

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
        model = DataParallel(model, use_chuncks=True)

    model.cuda()

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    else:
        bert_weight_name = json.load(open("config/bert_weight_name.json", "r"))
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        if args.from_pretrained:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

            # optimizer = torch.optim.Adamax(optimizer_grouped_parameters)

        else:
            optimizer = BertAdam(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                warmup=args.warmup_proportion,
                t_total=num_train_optimization_steps,
            )

    # lr_lambda = lambda x: lr_lambda_update(x)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = DataLoader(
            train_dset,
            shuffle=True,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        eval_dataloader = DataLoader(
            eval_dset,
            shuffle=True,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        loss_fct = CrossEntropyLoss(ignore_index=-1)

        startIterID = 0
        global_step = 0
        masked_loss_v_tmp = 0
        masked_loss_t_tmp = 0
        next_sentence_loss_tmp = 0
        loss_tmp = 0
        start_t = timer()
        num_options = 4
        model.train()

        for epochId in range(args.num_train_epochs):
            total_loss = 0
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            optimizer.zero_grad()
            train_score = 0

            for step, batch in enumerate(train_dataloader):
                iterId = startIterID + step + (epochId * len(train_dataloader))
                batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

                features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask = batch

                batch_size = features.size(0)
                max_num_bbox = features.size(1)

                features = features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 2048).contiguous().view(-1, max_num_bbox, 2048)
                spatials = spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 5).contiguous().view(-1, max_num_bbox, 5)
                image_mask = image_mask.unsqueeze(1).expand(batch_size, num_options, max_num_bbox).contiguous().view(-1, max_num_bbox)
                input_ids = input_ids.view(-1, input_ids.size(2))
                input_mask = input_mask.view(-1, input_mask.size(2))
                segment_ids = segment_ids.view(-1, segment_ids.size(2))
                co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))

                pred = model(input_ids, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)                
                pred = pred.view(batch_size, num_options)

                _, logits = torch.max(pred, 1)
                train_score += (logits == target).sum()
                loss = loss_fct(pred, target)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                viz.linePlot(iterId, loss.item(), "loss", "train")
                loss_tmp += loss.item()
                total_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps, args.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_this_step

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if step % 20 == 0 and step != 0:
                    loss_tmp = loss_tmp / 20.0

                    end_t = timer()
                    timeStamp = strftime("%a %d %b %y %X", gmtime())

                    Ep = epochId + nb_tr_steps / float(len(train_dataloader))
                    printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g]"

                    printInfo = [
                        timeStamp,
                        Ep,
                        nb_tr_steps,
                        end_t - start_t,
                        loss_tmp,
                    ]

                    start_t = end_t
                    print(printFormat % tuple(printInfo))

                    loss_tmp = 0

            train_score = 100 * train_score / len(train_dataloader.dataset)
            total_loss = total_loss / len(train_dataloader)

            model.eval()
            eval_loss, eval_score = evaluate(args, model, eval_dataloader)
            model.train()

            logger.info("epoch %d" % (epochId))
            logger.info("\ttrain_loss: %.2f, score: %.2f" % (total_loss, train_score))
            logger.info("\teval_loss: %.2f eval score: %.2f" % (eval_loss, 100 * eval_score))

            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            output_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + ".bin")
            if args.do_train:
                torch.save(model_to_save.state_dict(), output_model_file)


class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + "/" + exp_name
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)


def evaluate(args, model, dataloader):
    total_loss = 0
    num_data = 0
    score = 0
    foil_score = 0
    ori_score = 0
    foil_all = 0
    ori_all = 0
    loss_fct = CrossEntropyLoss(ignore_index=-1)
    for i, batch in enumerate(dataloader):
    # for batch in iter(dataloader):
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, input_ids, target, input_mask, segment_ids, co_attention_mask = batch
        batch_size = features.size(0)
        max_num_bbox = features.size(1)
        num_options = 4

        features = features.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 2048).contiguous().view(-1, max_num_bbox, 2048)
        spatials = spatials.unsqueeze(1).expand(batch_size, num_options, max_num_bbox, 5).contiguous().view(-1, max_num_bbox, 5)
        image_mask = image_mask.unsqueeze(1).expand(batch_size, num_options, max_num_bbox).contiguous().view(-1, max_num_bbox)
        input_ids = input_ids.view(-1, input_ids.size(2))
        input_mask = input_mask.view(-1, input_mask.size(2))
        segment_ids = segment_ids.view(-1, segment_ids.size(2))
        co_attention_mask = co_attention_mask.view(-1, co_attention_mask.size(2), co_attention_mask.size(3))
    
        with torch.no_grad():
            pred = model(input_ids, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)                
            pred = pred.view(batch_size, num_options)

        loss = loss_fct(pred, target)
        _, logits = torch.max(pred, 1)

        score += (logits == target).sum().item()

        total_loss += loss.item()
        num_data += pred.size(0)
        if i % 10 == 0:
            print(i, end=' ')

    return total_loss / float(len(dataloader)), score / float(num_data)

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