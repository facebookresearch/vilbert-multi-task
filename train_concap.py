# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import math
import sys

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

import vilbert.utils as utils
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig
import torch.distributed as dist


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--file_path",
        default="data/conceptual_caption/",
        type=str,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-base-uncased, roberta-base, roberta-large, ",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, roberta-base",
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
        type=str,
        default="config/bert_base_6layer_6conect.json",
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=36,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=512,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=10.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
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
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=25,
        help="Number of workers in the dataloader.",
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
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--without_coattention", action="store_true", help="whether pair loss."
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )

    parser.add_argument(
        "--objective",
        default=0,
        type=int,
        help="which objective to use \
        0: with ICA loss, \
        1: with ICA loss, for the not aligned pair, no masking objective, \
        2: without ICA loss, do not sample negative pair.",
    )
    parser.add_argument(
        "--num_negative", default=255, type=int, help="num of negative to use"
    )

    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )

    args = parser.parse_args()

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BertForMultiModalPreTraining
    else:
        from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig

    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""

    timeStamp = args.config_file.split("/")[1].split(".")[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(
        open("config/" + args.from_pretrained + "_weight_name.json", "r")
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

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)

    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    cache = 5000
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    num_train_optimization_steps = None
    train_dataset = ConceptCapLoaderTrain(
        args.file_path,
        tokenizer,
        args.bert_model,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        visual_target=args.visual_target,
        num_workers=args.num_workers,
        local_rank=args.local_rank,
        objective=args.objective,
        cache=cache,
    )

    validation_dataset = ConceptCapLoaderVal(
        args.file_path,
        tokenizer,
        args.bert_model,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        visual_target=args.visual_target,
        num_workers=2,
        objective=args.objective,
    )

    num_train_optimization_steps = int(
        train_dataset.num_dataset
        / args.train_batch_size
        / args.gradient_accumulation_steps
    ) * (args.num_train_epochs - args.start_epoch)

    task_names = ["Conceptual_Caption"]
    task_ids = ["TASK0"]
    task_num_iters = {"TASK0": train_dataset.num_dataset / args.train_batch_size}

    logdir = os.path.join("logs", timeStamp)
    if default_gpu:
        tbLogger = utils.tbLogger(
            logdir,
            savePath,
            task_names,
            task_ids,
            task_num_iters,
            args.gradient_accumulation_steps,
        )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if "roberta" in args.bert_model:
        config.model = "roberta"

    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False

    if args.dynamic_attention:
        config.dynamic_attention = True

    if args.from_pretrained:
        model = BertForMultiModalPreTraining.from_pretrained(
            args.from_pretrained, config=config, default_gpu=default_gpu
        )
    else:
        model = BertForMultiModalPreTraining(config)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

        if default_gpu:
            print(
                len(list(model.named_parameters())), len(optimizer_grouped_parameters)
            )

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

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=(0.9, 0.98),
        )

    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )

    startIterID = 0
    global_step = 0

    if args.resume_file != "" and os.path.exists(args.resume_file):
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        del checkpoint

    model.cuda()

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

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

    if default_gpu:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_dataset.num_dataset)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        for step, batch in enumerate(train_dataset):

            iterId = startIterID + step + (epochId * len(train_dataset))
            image_ids = batch[-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
                batch
            )

            if args.objective == 1:
                image_label = image_label * (is_next == 0).long().unsqueeze(1)
                image_label[image_label == 0] = -1

                lm_label_ids = lm_label_ids * (is_next == 0).long().unsqueeze(1)
                lm_label_ids[lm_label_ids == 0] = -1

            masked_loss_t, masked_loss_v, next_sentence_loss = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_target,
                is_next,
            )

            if args.objective == 2:
                next_sentence_loss = next_sentence_loss * 0

            masked_loss_v = masked_loss_v * args.img_weight
            loss = masked_loss_t + masked_loss_v + next_sentence_loss

            if n_gpu > 1:
                loss = loss.mean()
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                next_sentence_loss = next_sentence_loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / num_train_optimization_steps,
                        args.warmup_proportion,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if default_gpu:
                    tbLogger.step_train_CC(
                        epochId,
                        iterId,
                        float(masked_loss_t),
                        float(masked_loss_v),
                        float(next_sentence_loss),
                        optimizer.param_groups[0]["lr"],
                        "TASK0",
                        "train",
                    )

            if (
                step % (20 * args.gradient_accumulation_steps) == 0
                and step != 0
                and default_gpu
            ):
                tbLogger.showLossTrainCC()

        # Do the evaluation
        torch.set_grad_enabled(False)
        numBatches = len(validation_dataset)

        model.eval()
        for step, batch in enumerate(validation_dataset):
            image_ids = batch[-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
                batch
            )

            batch_size = input_ids.size(0)
            masked_loss_t, masked_loss_v, next_sentence_loss = model(
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
                lm_label_ids,
                image_label,
                image_target,
                is_next,
            )

            masked_loss_v = masked_loss_v * args.img_weight
            loss = masked_loss_t + masked_loss_v + next_sentence_loss

            if n_gpu > 1:
                loss = loss.mean()
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                next_sentence_loss = next_sentence_loss.mean()

            if default_gpu:
                tbLogger.step_val_CC(
                    epochId,
                    float(masked_loss_t),
                    float(masked_loss_v),
                    float(next_sentence_loss),
                    "TASK0",
                    batch_size,
                    "val",
                )
                sys.stdout.write("%d / %d \r" % (step, numBatches))
                sys.stdout.flush()

        if default_gpu:
            ave_score = tbLogger.showLossValCC()

        torch.set_grad_enabled(True)

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )
            output_checkpoint = os.path.join(
                savePath, "pytorch_ckpt_" + str(epochId) + ".tar"
            )
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                },
                output_checkpoint,
            )

    if default_gpu:
        tbLogger.txt_close()


if __name__ == "__main__":

    main()
