import h5py
import os
import pdb
import numpy as np
import json
import sys
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
import csv
import base64

csv.field_size_limit(sys.maxsize)

import sys
sys.path.append("tools/refer")


# data_root = '/srv/share2/jlu347/multi-modal-bert/data/referExpression'
# refer = REFER(data_root, dataset='refcoco+',  splitBy='unc')

# splits = ['train', 'val', 'test']
# for split in splits:
#     ref_ids = refer.getRefIds(split=split)
#     print('%s refs are in split [%s].' % (len(ref_ids), split))

# ref_ids = refer.getRefIds()
# ref_id = ref_ids[np.random.randint(0, len(ref_ids))]
# ref = refer.Refs[ref_id]
save_path = os.path.join('refcoco+.h5')
save_h5 = h5py.File(save_path, "w")

# count = 0
# num_file = 1
# name = '/srv/share2/jlu347/bottom-up-attention/feature/refcoco_unc/refcoco_unc_resnet101_faster_rcnn_genome.tsv.%d'
# infiles = [name % i for i in range(num_file)]

# for infile in infiles:
#     with open(infile) as tsv_in_file:
#         reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
#         for item in reader:
#             count += 1

count = 0
num_file = 1
name = '/srv/share2/jlu347/bottom-up-attention/feature/refcoco_unc/refcoco+_unc_resnet101_faster_rcnn_genome.tsv.%d'
infiles = [name % i for i in range(num_file)]
max_num_box = 0
for infile in infiles:
    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            num_boxes = int(item['num_boxes'])
            if max_num_box < num_boxes:
                max_num_box = num_boxes

            count += 1

            if count %1000 == 0:
                print(count)

print(max_num_box)
length = count

print("total length is %d" %length)
image_ids_h5d = save_h5.create_dataset(
    "image_ids", (length, )
)

image_w_h5d = save_h5.create_dataset(
    "image_w", (length, )
)

image_h_h5d = save_h5.create_dataset(
    "image_h", (length, )
)

num_boxes_h5d = save_h5.create_dataset(
    "num_boxes", (length, )
)

features_h5d = save_h5.create_dataset(
    "features", (length, max_num_box, 2048),
    chunks=(1, max_num_box, 2048)
)

cls_prob_h5d = save_h5.create_dataset(
    "cls_prob", (length, max_num_box, 1601),
    chunks=(1, max_num_box, 1601)
)

boxes_h5d = save_h5.create_dataset(
    "boxes", (length, max_num_box, 4),
    chunks=(1, max_num_box, 4)
)

count = 0
num_file = 1
name = '/srv/share2/jlu347/bottom-up-attention/feature/refcoco_unc/refcoco+_unc_resnet101_faster_rcnn_genome.tsv.%d'
infiles = [name % i for i in range(num_file)]

for infile in infiles:
    with open(infile) as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            
            image_id = item['image_id']
            image_h = item['image_h']
            image_w = item['image_w']
            num_boxes = item['num_boxes']
            boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
            features = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape(int(num_boxes), 2048)
            cls_prob = np.frombuffer(base64.b64decode(item['cls_prob']), dtype=np.float32).reshape(int(num_boxes), 1601)

            num_boxes = int(num_boxes)
            num_boxes_h5d[count] = int(num_boxes)

            image_ids_h5d[count] = int(image_id)
            image_w_h5d[count] = int(image_w)
            image_h_h5d[count] = int(image_h)

            boxes_h5d[count,:num_boxes] = boxes
            features_h5d[count,:num_boxes] = features
            cls_prob_h5d[count,:num_boxes] = cls_prob
    
            if count % 1000 == 0:
                print(count)
            count += 1

print(count)
save_h5.close()
