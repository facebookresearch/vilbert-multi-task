# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import numpy as np
import pdb
import argparse
from easydict import EasyDict as edict
import jsonlines
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="",
        type=str,
        help="",
    )    
    parser.add_argument(
        "--compared_file",
        default="",
        type=str,
        help="",
    )
    parser.add_argument(
        "--data_file",
        default="",
        type=str,
        help="",
    )
    parser.add_argument(
        "--tasks",
        default="",
        type=str,
        help="",
    )
    parser.add_argument(
        "--num_samples",
        default=100,
        type=int,
        help="",
    )
    args = parser.parse_args()
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    # 1: find the file where input file is correct while target file is wrong
    # 2: get the image id and image path.
    # 3: generate or copy the images. 
    # 4: Show based on different tasks. 
    #   - VQA: image, question, and different answers. 
    #   - RefCOCO: image, caption, and with different bounding box. 
    #   - VCR: image, question, answers. 
    #   - Retrieval: caption, and top 10 retrieved images. 

    input_file = json.load(open(args.input_file, 'r'))
    
    compared_file = None
    if args.compared_file != ''
        compared_file = json.load(open(args.compared_file, 'r'))

    task = 'TASK' + args.tasks
    task_names = task_cfg[task]['name']
    topK = 5
    if task in ['TASK9']:
        with jsonlines.open(args.data_file) as reader:
            entries = []
            imgid2entry = {}
            count = 0
            for annotation in reader:
                image_path = annotation['img_path']
                image_id = int(annotation['img_path'].split('.')[0])
                imgid2entry[image_id] = []
                for sentences in annotation['sentences']:
                    entries.append({"caption": sentences, 'image_id':image_id})
                    imgid2entry[image_id].append(count)
                    count += 1

        # identify which one is correct and which one is wrong.
        for i, seq in enumerate(input_file):
            correct_input = False
            target = int(i / 5)
            if target in seq[:topK]
                correct_input = True

        if compared_file:

        
        pdb.set_trace()

if __name__ == "__main__":
    main()

