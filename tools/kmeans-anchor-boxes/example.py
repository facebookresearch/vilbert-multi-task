import glob
import xml.etree.ElementTree as ET

import numpy as np
from kmeans import kmeans, avg_iou
import tensorpack.dataflow as td
import base64
import csv
import pdb
import os
import sys
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features', 'cls_prob']
csv.field_size_limit(sys.maxsize)


def load_dataset(path, num_examples):
    num_file = 30
    name = os.path.join(corpus_path, 'conceptual_caption_trainsubset_resnet101_faster_rcnn_genome.tsv.%d')
    infiles = [name % i for i in range(num_file)]
    dataset = []
    count = 0
    for infile in infiles:
        with open(infile) as tsv_in_file:
            reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                image_h = item['image_h']
                image_w = item['image_w']
                num_boxes = item['num_boxes']
                boxes = np.frombuffer(base64.b64decode(item['boxes']), dtype=np.float32).reshape(int(num_boxes), 4)
                boxes =copy.deepcopy(boxes)
                boxes[:,0] = boxes[:,0] / float(image_w)
                boxes[:,2] = boxes[:,2] / float(image_w)
                boxes[:,1] = boxes[:,1] / float(image_h)
                boxes[:,3] = boxes[:,3] / float(image_h)
                
                dataset.append(boxes)

                if count % 1000 == 0:
                    print(count)
                if count >= num_examples:
                    return np.concatenate(dataset, axis=0)
                count += 1

num_caps = 800000
corpus_path = '/srv/share2/jlu347/bottom-up-attention/feature/conceptual_caption'
CLUSTERS = 300


# data = load_dataset(corpus_path, num_caps)
# np.save(open('bboxs.npy', 'wb'), data)
data = np.load(open('bboxs.npy', 'rb'))
idx = np.random.randint(len(data), size=num_caps)
data = data[idx]
out = kmeans(data, k=CLUSTERS)
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
print("Boxes:\n {}".format(out))
np.save(open('centers_'+ str(CLUSTERS) + '.npy', 'wb'), out.cpu().numpy())

fig,ax = plt.subplots(1)
rect = patches.Rectangle((50,100),40,30,linewidth=1,edgecolor='r',facecolor='none')
ax.add_patch(rect)
plt.savefig('test.jpg')

pdb.set_trace()

