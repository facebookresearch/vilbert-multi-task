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

from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig
import torch.distributed as dist

import pdb

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
        default="/srv/share/vgoswami8/conceptual_captions",
        type=str,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--bert_model",
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
        type=str,
        required=True,
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
    parser.add_argument("--predict_feature", action="store_true", help="visual target.")
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
        "--dynamic_attention", action="store_true" , help="whether use dynamic attention."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=3,
        help="Number of workers in the dataloader.",
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
    parser.add_argument(
        "--freeze", default = -1, type=int, 
        help="till which layer of textual stream of vilbert need to fixed."
    )
    parser.add_argument(
        "--distributed", action="store_true" , help="whether use chunck for parallel training."
    )
    parser.add_argument(
        "--without_coattention", action="store_true" , help="whether pair loss."
    )
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")

    args = parser.parse_args()
    if args.baseline:
        from pytorch_pretrained_bert.modeling import BertConfig
        from vilbert.basebert import BertForMultiModalPreTraining
    else:
        from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig

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
    
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False
    
    if args.dynamic_attention:
        config.dynamic_attention = True
        
    # save all the hidden parameters. 
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    bert_weight_name = json.load(open("config/" + args.from_pretrained + "_weight_name.json", "r"))
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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    num_train_optimization_steps = None
    viz = TBlogger("logs", timeStamp)

    train_dataset = ConceptCapLoaderTrain(
        args.file_path,
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=args.num_workers,
        distributed=args.distributed,
    )
    
    validation_dataset = ConceptCapLoaderVal(
        args.file_path,
        tokenizer,
        seq_len=args.max_seq_length,
        batch_size=args.train_batch_size,
        predict_feature=args.predict_feature,
        num_workers=2,
        distributed=args.distributed,
    )

    num_train_optimization_steps = (
        int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        )
        * (args.num_train_epochs - args.start_epoch)
    )
    # if args.local_rank != -1:
    #     num_train_optimization_steps = (
    #         num_train_optimization_steps // torch.distributed.get_world_size()
    #     )

    default_gpu = False
    if dist.is_available() and args.distributed:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    # pdb.set_trace()
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.from_pretrained:
        model = BertForMultiModalPreTraining.from_pretrained(args.from_pretrained, config)
    else:
        model = BertForMultiModalPreTraining(config)

    model.cuda()

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
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
        if default_gpu:
            print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion*num_train_optimization_steps, t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", train_dataset.num_dataset)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    startIterID = 0
    global_step = 0
    masked_loss_v_tmp = 0
    masked_loss_t_tmp = 0
    next_sentence_loss_tmp = 0
    loss_tmp = 0
    start_t = timer()

    for epochId in range(int(args.start_epoch), int(args.num_train_epochs)):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(train_dataset):
            iterId = startIterID + step + (epochId * len(train_dataset))
            image_ids = batch[-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
                batch
            )

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

            if args.without_coattention:
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
    
            tr_loss += loss.item()

            rank = 0
            if dist.is_available() and args.distributed:
                rank = dist.get_rank()
            else:
                rank = 0
                
            viz.linePlot(iterId, loss.item(), "loss_"+str(rank), "train")
            viz.linePlot(iterId, masked_loss_t.item(), "masked_loss_t_"+str(rank), "train")
            viz.linePlot(iterId, masked_loss_v.item(), "masked_loss_v_"+str(rank), "train")
            viz.linePlot(
                iterId, next_sentence_loss.item(), "next_sentence_loss_"+str(rank), "train"
            )
            # viz.linePlot(iterId, optimizer.get_lr()[0], 'learning_rate', 'train')

            loss_tmp += loss.item()
            masked_loss_v_tmp += masked_loss_v.item()
            masked_loss_t_tmp += masked_loss_t.item()
            next_sentence_loss_tmp += next_sentence_loss.item()

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
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

            if step % 20 == 0 and step != 0:
                masked_loss_t_tmp = masked_loss_t_tmp / 20.0
                masked_loss_v_tmp = masked_loss_v_tmp / 20.0
                next_sentence_loss_tmp = next_sentence_loss_tmp / 20.0
                loss_tmp = loss_tmp / 20.0

                end_t = timer()
                timeStamp = strftime("%a %d %b %y %X", gmtime())

                Ep = epochId + nb_tr_steps / float(len(train_dataset))
                printFormat = "[%s][Ep: %.2f][Iter: %d][Time: %5.2fs][Loss: %.5g][Loss_v: %.5g][Loss_t: %.5g][Loss_n: %.5g][LR: %.8g]"

                printInfo = [
                    timeStamp,
                    Ep,
                    nb_tr_steps,
                    end_t - start_t,
                    loss_tmp,
                    masked_loss_v_tmp,
                    masked_loss_t_tmp,
                    next_sentence_loss_tmp,
                    0,
                    # optimizer.get_lr()[0],
                ]
                
                start_t = end_t
                print(printFormat % tuple(printInfo))

                masked_loss_v_tmp = 0
                masked_loss_t_tmp = 0
                next_sentence_loss_tmp = 0
                loss_tmp = 0

        # Do the evaluation 
        torch.set_grad_enabled(False)    
        start_t = timer()
        numBatches = len(validation_dataset)
        eval_masked_loss_t = 0
        eval_masked_loss_v = 0
        eval_next_sentence_loss = 0
        eval_total_loss = 0

        model.eval()
        for step, batch in enumerate(validation_dataset):

            image_ids = batch[-1]
            batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
                batch
            )

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
                loss = loss.mean()  # mean() to average on multi-gpu.
                masked_loss_t = masked_loss_t.mean()
                masked_loss_v = masked_loss_v.mean()
                next_sentence_loss = next_sentence_loss.mean()

            eval_masked_loss_t += masked_loss_t.item()
            eval_masked_loss_v += masked_loss_v.item()
            eval_next_sentence_loss += next_sentence_loss.item()
            eval_total_loss += loss.item()

            end_t = timer()
            delta_t = " Time: %5.2fs" % (end_t - start_t)
            start_t = end_t
            progressString = "\r Evaluating split '%s' [%d/%d]\t" + delta_t
            sys.stdout.write(progressString % ('val', step + 1, numBatches))
            sys.stdout.flush()

        eval_masked_loss_t = eval_masked_loss_t / float(numBatches)
        eval_masked_loss_v = eval_masked_loss_v / float(numBatches)
        eval_next_sentence_loss = eval_next_sentence_loss / float(numBatches)
        eval_total_loss = eval_total_loss / float(numBatches)

        printFormat = "Evaluation: [Loss: %.5g][Loss_v: %.5g][Loss_t: %.5g][Loss_n: %.5g]"
        printInfo = [
            eval_total_loss,
            eval_masked_loss_v,
            eval_masked_loss_t,
            eval_next_sentence_loss]

        print(printFormat % tuple(printInfo))
        torch.set_grad_enabled(True)   

        viz.linePlot(epochId, eval_total_loss, "loss_" + str(rank), "val")
        viz.linePlot(epochId, eval_masked_loss_t, "masked_loss_t_" + str(rank), "val")
        viz.linePlot(epochId, eval_masked_loss_v, "masked_loss_v_" + str(rank), "val")
        viz.linePlot(epochId, eval_next_sentence_loss, "next_sentence_loss_" + str(rank), "val")

        if default_gpu:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )

            torch.save(model_to_save.state_dict(), output_model_file)

class TBlogger:
    def __init__(self, log_dir, exp_name):
        log_dir = log_dir + "/" + exp_name
        print("logging file at: " + log_dir)
        self.logger = SummaryWriter(log_dir=log_dir)

    def linePlot(self, step, val, split, key, xlabel="None"):
        self.logger.add_scalar(split + "/" + key, val, step)

if __name__ == "__main__":

    main()
