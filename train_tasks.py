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

from pytorch_transformers.optimization import AdamW, WarmupConstantSchedule, WarmupLinearSchedule

from vilbert.optimization import RAdam
from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

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
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_iter_multiplier",
        default=1.0,
        type=float,
        help="multiplier for the multi-task training.",
    )
    parser.add_argument(
        "--train_iter_gap",
        default=4,
        type=int,
        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
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
        "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
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
        "--optim", default='AdamW', type=str, help="what to use for the optimization."
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
    parser.add_argument(
        "--clean_train_sets", default=True , type=bool, help="whether clean train sets for multitask data."
    )
    parser.add_argument(
        "--visual_target", default=0, type=int, 
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.")
    parser.add_argument(
        "--task_specific_tokens", action="store_true", 
        help="whether to use task specific tokens for the multi-task learning.")

    args = parser.parse_args()
    with open('vilbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
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
    
    logdir = os.path.join(savePath, 'logs')
    tbLogger = utils.tbLogger(logdir, savePath, task_names, task_ids, task_num_iters, args.gradient_accumulation_steps)

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_ave_iter = {}
    task_stop_controller = {}
    for task_id, num_iter in task_num_iters.items(): 
        task_ave_iter[task_id] = int(task_cfg[task]['num_epoch'] * num_iter * args.train_iter_multiplier / args.num_train_epochs)
        task_stop_controller[task_id] = utils.MultiTaskStopOnPlateau(mode='max', patience=1, continue_threshold=0.005, cooldown=1, threshold=0.001)

    task_ave_iter_list = sorted(task_ave_iter.values())
    median_num_iter = task_ave_iter_list[-1]
    num_train_optimization_steps = median_num_iter * args.num_train_epochs // args.gradient_accumulation_steps
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    if args.dynamic_attention:
        config.dynamic_attention = True
    if 'roberta' in args.bert_model:
        config.model = 'roberta'

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )

    task_losses = LoadLosses(args, task_cfg, args.tasks.split('-'))

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
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'vil_' in key:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = base_lr
                    else:
                        lr = 1e-4
                else:
                    lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    
    if args.optim == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False) 
    elif args.optim == 'RAdam':
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr)

    warmpu_steps = args.warmup_proportion*num_train_optimization_steps

    if args.lr_scheduler == 'warmup_linear': 
        warmup_scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmpu_steps, t_total=num_train_optimization_steps)
    else:
        warmup_scheduler = WarmupConstantSchedule(optimizer, warmup_steps=warmpu_steps)

    lr_reduce_list = np.array([5, 7])
    if args.lr_scheduler == 'automatic':
        lr_scheduler = ReduceLROnPlateau(optimizer, \
                        mode='max',
                        factor=0.2, 
                        patience=1,
                        cooldown=1,
                        threshold=0.001)
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=median_num_iter*args.num_train_epochs)
    elif args.lr_scheduler == 'cosine_warm':
        lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=median_num_iter*args.num_train_epochs)
    elif args.lr_scheduler == 'mannul':
        def lr_lambda_fun(epoch):
            return pow(0.2, np.sum(lr_reduce_list <= epoch))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

    startIterID = 0
    global_step = 0
    start_epoch = 0

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
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        global_step = checkpoint['global_step']
        start_epoch = int(checkpoint['epoch_id']) + 1
        task_stop_controller = checkpoint['task_stop_controller']
        tbLogger = checkpoint['tb_logger']
        del checkpoint

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

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

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)
        print("  Num steps: %d" %num_train_optimization_steps)

    task_iter_train = {name:None for name in task_ids}
    task_count = {name:0 for name in task_ids}
    for epochId in tqdm(range(start_epoch, args.num_train_epochs), desc="Epoch"):
        model.train()
        for step in range(median_num_iter):
            iterId = startIterID + step + (epochId * median_num_iter)
            first_task = True
            for task_id in task_ids:                
                is_forward = False
                if (not task_stop_controller[task_id].in_stop) or (iterId % args.train_iter_gap == 0):
                    is_forward = True

                if is_forward:
                    loss, score = ForwardModelsTrain(args, task_cfg, device, task_id, task_count, \
                                task_iter_train, task_dataloader_train, model, task_losses)

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

                        if first_task and (global_step < warmpu_steps or args.lr_scheduler == 'warmup_linear'):
                            warmup_scheduler.step()
                        
                        optimizer.step()
                        model.zero_grad()
                        if first_task:
                            global_step += 1
                            first_task = False
                            
                        if default_gpu:
                            tbLogger.step_train(epochId, iterId, float(loss), float(score), \
                                                    optimizer.param_groups[0]['lr'], task_id, 'train')

            if 'cosine' in args.lr_scheduler and global_step > warmpu_steps:
                lr_scheduler.step()

            if step % (20 * args.gradient_accumulation_steps) == 0 and step != 0 and default_gpu:
                tbLogger.showLossTrain()        

            # decided whether to evaluate on each tasks.
            for task_id in task_ids:
                if (iterId!= 0 and iterId % task_num_iters[task_id] == 0) or (epochId == args.num_train_epochs - 1 and step == median_num_iter-1):
                    evaluate(args, task_dataloader_val, task_stop_controller, task_cfg, \
                                device, task_id, model, task_losses, epochId, default_gpu, tbLogger)

        if args.lr_scheduler == 'automatic':
            lr_scheduler.step(sum(val_scores.values()))
            logger.info("best average score is %3f" %lr_scheduler.best)
        elif args.lr_scheduler == 'mannul':
            lr_scheduler.step()

        if epochId in lr_reduce_list: 
            for task_id in task_ids:
                # reset the task_stop_controller once the lr drop
                task_stop_controller[task_id]._reset()

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
                savePath, "pytorch_ckpt_latest.tar"
            )
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save({
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
                # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'global_step': global_step,
                'epoch_id': epochId,
                'task_stop_controller': task_stop_controller,
                'tb_logger': tbLogger,
            }, output_checkpoint)
    tbLogger.txt_close()


def evaluate(args, task_dataloader_val, task_stop_controller, task_cfg, device, task_id, model, task_losses, epochId, default_gpu, tbLogger):
    
    model.eval()
    for i, batch in enumerate(task_dataloader_val[task_id]):
        loss, score, batch_size = ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses)
        tbLogger.step_val(epochId, float(loss), float(score), task_id, batch_size, 'val')
        if default_gpu:
            sys.stdout.write('%d/%d\r' % (i, len(task_dataloader_val[task_id])))
            sys.stdout.flush()

    # update the multi-task scheduler.
    task_stop_controller[task_id].step(tbLogger.getValScore(task_id))
    score = tbLogger.showLossVal(task_id, task_stop_controller)
    model.train()

if __name__ == "__main__":

    main()
