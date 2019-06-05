import argparse
import json
import logging
import os
import random
from io import open
import numpy as np

from time import gmtime, strftime
from timeit import default_timer as timer

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from parallel.parallel import DataParallelModel, DataParallelCriterion
from vilbert.vilbert import BertConfig
from vilbert.task_utils import LoadDatasets, LoadModels, ForwardModels
from vilbert.vilbert import MultiModalBertForVLClassifier,MultiModalBertForVLLogit, \
                            MultiModalBertForVLogit, MultiModalBertForAllTasks


ModelMap = {'ALL-tasks': MultiModalBertForAllTasks,
            'VL-classifier': MultiModalBertForVLClassifier,
            'VL-logit': MultiModalBertForVLLogit,
            'V-logit': MultiModalBertForVLogit,
            }

LossMap = {'BCEWithLogitLoss': nn.BCEWithLogitsLoss(reduction='mean'),
                }

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
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
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
        "--save_name",
        default='',
        type=str,
        help="save name for training.", 
    )
    parser.add_argument(
        "--use_chunk", default=0, type=float, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--in_memory", default=True, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='adam', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )

    args = parser.parse_args()
    with open('config/vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    task_ids = args.tasks.split('-')
    task_batch_size, task_num_iters, task_names, task_datasets_train, task_datasets_val, \
            task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, task_ids)

    timeStamp = '-'.join(task_names) + '_' + args.bert_model 
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

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = None
    viz = TBlogger("logs/" + timeStamp)

    num_train_optimization_steps = (
        max([len(dataloader) for dataloader in task_dataloader_train.values()])* args.num_train_epochs
    )

    if args.local_rank != -1:
        num_train_optimization_steps = (
            num_train_optimization_steps // torch.distributed.get_world_size()
        )

    model, loss_funs = LoadModels(args, task_cfg, task_ids, task_datasets_train, ModelMap, LossMap, config)

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
        model = DataParallelModel(model)
        criterion = DataParallelCriterion(criterion)

    model.cuda()

    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

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
            warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                 t_total=num_train_optimization_steps)
    else:
        if args.from_pretrained:
            if args.optimizer == 'adam':
                optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
            elif args.optimizer == 'admax':
                optimizer = torch.optim.Adamax(optimizer_grouped_parameters)
        else:
            if args.optimizer == 'adam':
                optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=args.learning_rate,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)
            elif args.optimizer == 'admax':
                optimizer = torch.optim.Adamax(optimizer_grouped_parameters)
    
    if args.optimizer == 'admax':
        lr_lambda = lambda x: lr_lambda_update(x)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info("***** Running training *****")
    # logger.info("  Num examples = %d", len(train_dset))
    # logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    loss_tmp = 0
    max_num_iter = max(task_num_iters.values())
    start_t = timer()
    model.train()
    # t1 = timer()

    # initialize the data iteration.

    task_iter_train = {name:iter(dataloader) for name, dataloader in task_dataloader_train.items()}
    task_count = {name:0 for name in task_names}
    for epochId in tqdm(range(args.num_train_epochs), desc="Epoch"):
        total_loss = 0
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_score = 0
        optimizer.zero_grad()
        for step in range(max_num_iter):
            iterId = startIterID + step + (epochId * max_num_iter)
            for task_name in task_names:
                task_count[task_name] += 1
                if iterId % task_count[task_name] == 0:
                    task_iter_train[task_name] = iter(task_dataloader_train[task_name])
                
                batch = task_iter_train[task_name].next()
                batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)
                features, spatials, image_mask, question, target, input_mask, segment_ids, question_ids = batch
                
                preds = model(question, features, spatials, segment_ids, input_mask, image_mask)
                loss = loss_funs[task_name](preds, target)
                if n_gpu > 1:
                    loss = loss.mean()
                loss = loss * target.size(1)
                
                preds_gathered = torch.cat([pred.detach() for pred in preds], dim=0)
                batch_score = compute_score_with_logits(preds_gathered, target).sum()

                # nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                total_loss += loss.item() * features.size(0)
                train_score += batch_score

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                # print(loss)
                # print(tr_loss)
                viz.linePlot(iterId, loss.item(), "loss", "train")
                # viz.linePlot(iterId, optimizer.get_lr()[0], 'learning_rate', 'train')
                loss_tmp += loss.item()

                nb_tr_examples += question.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps, args.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_this_step
        
                    # nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if args.optimizer == 'admax':
                    lr_scheduler.step(iterId)

                if step % 20 == 0 and step != 0:
                    loss_tmp = loss_tmp / 20.0

                    end_t = timer()
                    timeStamp = strftime("%a %d %b %y %X", gmtime())

                    Ep = epochId + nb_tr_steps / float(len(train_dataloader))
                    printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g]"

                    printInfo = [
                        timeStamp,
                        Ep,
                        iterId,
                        end_t - start_t,
                        loss_tmp,
                    ]

                    start_t = end_t
                    print(printFormat % tuple(printInfo))

                    loss_tmp = 0

        model.train(False)
        eval_score, bound = evaluate(args, model, eval_dataloader)
        model.train(True)

        train_score = 100 * train_score / len(train_dataloader.dataset)
        total_loss = total_loss / len(train_dataloader.dataset)
        logger.info("epoch %d" % (epochId))
        logger.info("\ttrain_loss: %.2f, score: %.2f" % (total_loss, train_score))
        logger.info("\teval score: %.2f (%.2f)" % (100 * eval_score, 100 * bound))

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
    def __init__(self, log_dir):
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

def evaluate(args, model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    for batch in dataloader:
        batch = tuple(t.cuda() for t in batch)
        features, spatials, image_mask, question, target, input_mask, segment_ids, question_ids = batch

        with torch.no_grad():
            preds = model(question, features, spatials, segment_ids, input_mask, image_mask)
            preds_gathered = torch.cat([pred.detach() for pred in preds], dim=0)
            batch_score = compute_score_with_logits(preds_gathered, target.cuda()).sum()
            score += batch_score.item()
            upper_bound += (target.max(1)[0]).sum()
            num_data += features.size(0)


    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound

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
