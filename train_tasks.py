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

from pytorch_pretrained_bert.optimization import WarmupLinearSchedule

from parallel.parallel import DataParallelModel, DataParallelCriterion
from vilbert.vilbert import BertConfig
from vilbert.task_utils import LoadDatasets, LoadLosses, ForwardModelsTrain, ForwardModelsVal
from vilbert.vilbert import VILBertForVLTasks
from vilbert.optimization import BertAdam, VqaAdam, adjust_lr
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
        "--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=30,
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
        "--num_workers", type=int, default=16, help="Number of workers in the dataloader."
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
        "--in_memory", default=False, type=bool, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--optimizer", default='adam', type=str, help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--tasks", default='', type=str, help="1-2-3... training task separate by -"
    )
    parser.add_argument(
        "--freeze", default = -1, type=int, 
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument("--predict_feature", action="store_true", help="visual target.")
    parser.add_argument("--vision_pretrained", action="store_true", help="whether pre-trained the image or not.")

    args = parser.parse_args()
    with open('config/vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

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
        with open(os.path.join(savePath, 'command.txt'), 'w') as f:
            print(args, file=f)  # Python 3.x
            print('\n', file=f)
            print(config, file=f)

    task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, \
            task_dataloader_train, task_dataloader_val = LoadDatasets(args, task_cfg, args.tasks.split('-'))

    tbLogger = utils.tbLogger(timeStamp, task_names, task_ids, task_num_iters)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = max(task_num_iters.values())* args.num_train_epochs
    num_labels = max([dataset.num_labels for dataset in task_datasets_train.values()])

    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config, num_labels=num_labels
        )
    else:
        model = VILBertForVLTasks(
            args.bert_model, config, num_labels=num_labels)

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
        model = DataParallelModel(model)

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

    if args.vision_pretrained:
        optimizer_grouped_parameters = []
        lr = args.learning_rate
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                optimizer_grouped_parameters += [{"params": [value], "lr": lr}]

        # param_optimizer = list(model.named_parameters())
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.01,
        #     },
        #     {
        #         "params": [
        #             p for n, p in param_optimizer if any(nd in n for nd in no_decay)
        #         ],
        #         "weight_decay": 0.0,
        #     },
        # ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.2
                else:
                    lr = args.learning_rate

                optimizer_grouped_parameters += [{"params": [value], "lr": lr}]

                # if any(nd in key for nd in no_decay):
                #     optimizer_grouped_parameters += [
                #         {"params": [value], "lr": lr, "weight_decay": 0.01}
                #     ]

                # if not any(nd in key for nd in no_decay):
                #     optimizer_grouped_parameters += [
                #         {"params": [value], "lr": lr, "weight_decay": 0.0}
                #     ]

    # optimizer_grouped_parameters = []
    # for key, value in dict(model.named_parameters()).items():
    #     if value.requires_grad:
    #         optimizer_grouped_parameters += [{"params": [value]}]
    
    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))
    max_num_iter = max(task_num_iters.values())
    max_batch_size = max(task_batch_size.values())
    if args.optimizer == 'adam':
        optim = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    elif args.optimizer == 'VqaAdam':
        optim = VqaAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         data_size= max_batch_size * max_num_iter, 
                         batch_size=max_batch_size)
    else:
        assert False 

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", task_batch_size)    
        print("  Num steps: %d" %num_train_optimization_steps)

    startIterID = 0
    # initialize the data iteration.
    task_iter_train = {name:None for name in task_ids}
    task_count = {name:0 for name in task_ids}
    for epochId in tqdm(range(args.num_train_epochs), desc="Epoch"):

        if args.optimizer == 'VqaAdam':
            if epochId in [10, 12]:
                adjust_lr(optim, 0.2)

        model.train()
        for step in range(max_num_iter):
            iterId = startIterID + step + (epochId * max_num_iter)
            for task_id in task_ids:
                loss, score = ForwardModelsTrain(args, task_cfg, device, task_id, iterId, task_count, task_iter_train, task_dataloader_train, model, task_losses)
                optim.zero_grad()
                loss.backward()
                optim.step()

                if default_gpu:
                    tbLogger.step_train(epochId, iterId, float(loss), float(score), optim.rate(), task_id, 'train')

            if step % 20 == 0 and default_gpu:
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

        if default_gpu: tbLogger.showLossVal()

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self

            if not os.path.exists(savePath):
                os.makedirs(savePath)
            output_model_file = os.path.join(savePath, "pytorch_model_" + str(epochId) + ".bin")
            
            model_save = {}
            model_save['model'] = model_to_save.state_dict()
            model_save['optim'] = optim
            model_save['epoch'] = epochId
            torch.save(model_save, output_model_file)

if __name__ == "__main__":

    main()
