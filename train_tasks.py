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

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule

from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

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
        "--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
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
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
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
        "--num_workers", type=int, default=24, help="Number of workers in the dataloader."
    )
    parser.add_argument(
        "--save_name",
        default='',
        type=str,
        help="save name for training.", 
    )
    parser.add_argument(
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='AdamW', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze", default = -1, type=int, 
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--vision_scratch", action="store_true", help="whether pre-trained the image or not."
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler", default='mannul', type=str, help="whether use learning rate scheduler."
    )  
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument("--resume_file", 
        default="",
        type=str,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dynamic_attention", action="store_true" , help="whether use dynamic attention."
    )

    args = parser.parse_args()
    with open('vilbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BaseBertForVLTasks      
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    task_lr = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)
        task_lr.append(task_cfg[task]['lr'])

    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name:
        prefix = '-' + args.save_name
    else:
        prefix = ''
    timeStamp = '-'.join(task_names) + '_' + args.config_file.split('/')[1].split('.')[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
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
        with open(os.path.join(savePath, 'command.txt'), 'w') as f:
            print(args, file=f)  # Python 3.x
            print('\n', file=f)
            print(config, file=f)

    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, \
            task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split('-'))
    
    logdir = os.path.join('logs', timeStamp)
    tbLogger = utils.tbLogger(logdir, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = max(task_num_iters.values()) * args.num_train_epochs // args.gradient_accumulation_steps
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    task_start_iter = {}
    task_interval = {}
    for task_id, num_iter in task_num_iters.items():
        task_start_iter[task_id] = num_train_optimization_steps - (task_cfg[task]['num_epoch'] * num_iter // args.gradient_accumulation_steps)
        task_interval[task_id] = num_train_optimization_steps // (task_cfg[task]['num_epoch'] * num_iter // args.gradient_accumulation_steps)

    if args.dynamic_attention:
        config.dynamic_attention = True

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split('-'))
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if 'embeddings' in name:
                bert_weight_name_filtered.append(name)
            elif 'encoder' in name:
                layer_num = name.split('.')[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)
        
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False
            
        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    optimizer_grouped_parameters = []
    lr = args.learning_rate
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_prediction' in key:
                # if args.learning_rate <= 2e-5:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = args.learning_rate
                    else:
                        lr = 1e-4
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

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    max_num_iter = max(task_num_iters.values())
    max_batch_size = max(task_batch_size.values())
    
    if args.optimizer == 'AdamW':    
        optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    warmpu_steps = args.warmup_proportion*num_train_optimization_steps
    warmup_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmpu_steps)

    if args.lr_scheduler == 'automatic':
        lr_scheduler = ReduceLROnPlateau(optimizer, \
                        mode='max',
                        factor=0.2, 
                        patience=1,
                        cooldown=1,
                        threshold=0.001)
    elif args.lr_scheduler == 'mannul':
        lr_reduce_list = np.array([6, 8])
        def lr_lambda_fun(epoch):
            return pow(0.1, np.sum(lr_reduce_list <= epoch))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" %num_train_optimization_steps)

    startIterID = 0
    global_step = 0

    if args.resume_file != "" and os.path.exists(args.resume_file):
        checkpoint = torch.load(args.resume_file, map_location='cpu')
        new_dict = {}
        for attr in checkpoint['model_state_dict']:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint['model_state_dict'][attr]
            else:
                new_dict[attr] = checkpoint['model_state_dict'][attr]
        model.load_state_dict(new_dict)
        warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        del checkpoint

    # initialize the data iteration.
    task_iter_train = {name:None for name in task_ids}
    task_count = {name:0 for name in task_ids}
    for epochId in tqdm(range(args.num_train_epochs), desc="Epoch"):
        model.train()
        for step in range(max_num_iter):
            iterId = startIterID + step + (epochId * max_num_iter)
            for task_id in task_ids:
                if iterId >= task_start_iter[task_id]:
                # if iterId % task_interval[task_id] == 0:
                    loss, score = ForwardModelsTrain(args, task_cfg, device, task_id, task_count, task_iter_train, task_dataloader_train, model, task_losses, task_start_iter)
                    loss = loss * loss_scale[task_id]
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps 

                    loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = args.learning_rate * warmup_linear(
                                global_step / num_train_optimization_steps,
                                args.warmup_proportion,
                            )
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = lr_this_step

                        if global_step < warmpu_steps:
                            warmup_scheduler.step()
                        
                        optimizer.step()
                        model.zero_grad()
                        global_step += 1
                        
                        if default_gpu:
                            tbLogger.step_train(epochId, iterId, float(loss), float(score), optimizer.param_groups[0]['lr'], task_id, 'train')

            if step % (20 * args.gradient_accumulation_steps) == 0 and step != 0 and default_gpu:
                tbLogger.showLossTrain()

        model.eval()
        # when run evaluate, we run each task sequentially. 
        for task_id in task_ids:
            for i, batch in enumerate(task_dataloader_val[task_id]):
                loss, score, batch_size = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses)
                tbLogger.step_val(epochId, float(loss), float(score), task_id, batch_size, 'val')
                if default_gpu:
                    sys.stdout.write('%d/%d\r' % (i, len(task_dataloader_val[task_id])))
                    sys.stdout.flush()
        
        ave_score = tbLogger.showLossVal()
        if args.lr_scheduler == 'automatic':
            lr_scheduler.step(ave_score)
            logger.info("best average score is %3f" %lr_scheduler.best)
        else:
            lr_scheduler.step()

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
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'global_step': global_step,

            }, output_checkpoint)

    tbLogger.txt_close()
    
if __name__ == "__main__":

    main()