import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from _image_features_reader import ImageFeaturesH5Reader
from vqa_dataset import VQAClassificationDataset
from dataset import Dictionary, VQAFeatureDataset, BertDictionary, BertFeatureDataset
import base_model
from train import train
import utils
from pytorch_pretrained_bert.tokenization import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--bert', type=bool, default=True, help='use bert pretrained model')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--bert_model", default='bert-base-uncased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.bert:

        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
        image_features_reader_train = ImageFeaturesH5Reader('data/COCO/COCO_train.h5', True)
        image_features_reader_val = ImageFeaturesH5Reader('data/COCO/COCO_validation.h5', True)

        train_dset = VQAClassificationDataset(
            "train", image_features_reader_train, tokenizer, dataroot="data/VQA"
        )
        eval_dset = VQAClassificationDataset("val", image_features_reader_val, tokenizer, dataroot="data/VQA")

        # dictionary = BertDictionary(args)        
        # train_dset = BertFeatureDataset('train', dictionary, dataroot='data/VQA')
        # eval_dset = BertFeatureDataset('val', dictionary, dataroot='data/VQA')
    else:
        pass
        # dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        # train_dset = VQAFeatureDataset('train', dictionary)
        # eval_dset = VQAFeatureDataset('val', dictionary)
    batch_size = args.batch_size

    if args.bert:
        constructor = 'build_%s_bert' % args.model    
    else:
        constructor = 'build_%s' % args.model
    

    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    
    if not args.bert:
        model.w_emb.init_embedding('data/glove6b_init_300d.npy')

    model = nn.DataParallel(model).cuda()

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=10)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=10)
    train(args, model, train_loader, eval_loader, args.epochs, args.output)
