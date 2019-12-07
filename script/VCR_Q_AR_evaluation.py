# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pdb
import numpy as np
import pandas as pd
import sys
import json_lines


print(sys.argv)

json_path_1 = sys.argv[1]
json_path_2 = sys.argv[2]

qa_result = json.load(open(json_path_1, "r"))
qar_result = json.load(open(json_path_2, "r"))

num = len(qa_result)
print(num)

annotations_jsonpath = "data/VCR/val.jsonl"
ground_truth = []
# load the file.
with open(annotations_jsonpath, "rb") as f:  # opening file in binary(rb) mode
    for annotation in json_lines.reader(f):
        answer_label = annotation["answer_label"]
        rationale_label = annotation["rationale_label"]
        anno_id = int(annotation["annot_id"].split("-")[1])
        ground_truth.append(
            {
                "answer_label": answer_label,
                "rationale_label": rationale_label,
                "anno_id": anno_id,
            }
        )

Q_A_accuracy = 0
QA_R_accuracy = 0
Q_AR_accuracy = 0
for i in range(num):
    answer = np.argmax(qa_result[i]["answer"])
    gt_answer = ground_truth[i]["answer_label"]

    rationale = np.argmax(qar_result[i]["answer"])
    gt_rationale = ground_truth[i]["rationale_label"]

    if answer == gt_answer:
        Q_A_accuracy += 1
    if rationale == gt_rationale:
        QA_R_accuracy += 1
    if answer == gt_answer and rationale == gt_rationale:
        Q_AR_accuracy += 1


Q_A_accuracy = Q_A_accuracy / float(num)
QA_R_accuracy = QA_R_accuracy / float(num)
Q_AR_accuracy = Q_AR_accuracy / float(num)

print("Q_A_accuracy: %f" % Q_A_accuracy)
print("QA_R_accuracy: %f" % QA_R_accuracy)
print("Q_AR_accuracy: %f" % Q_AR_accuracy)
