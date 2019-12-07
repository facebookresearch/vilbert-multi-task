# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pdb
import numpy as np
import pandas as pd


json_path_1 = "results/VCR_Q-A-VCR_QA-R_bert_base_6layer_6conect-Q_A/test_result.json"
json_path_2 = "results/VCR_Q-A-VCR_QA-R_bert_base_6layer_6conect-Q_AR/test_result.json"
output_path = "results/vcr_submission_vilbert.csv"

qa_result = json.load(open(json_path_1, "r"))
qar_result = json.load(open(json_path_2, "r"))

num = len(qa_result)
probs_grp = np.zeros([num, 5, 4])

ids_grp = []
for i in range(num):
    tmp = []
    tmp.append(qa_result[i]["answer"])
    for j in range(4):
        tmp.append(qar_result[i * 4 + j]["answer"])
    probs_grp[i] = np.array(tmp)
    ids_grp.append("test-%d" % qa_result[i]["question_id"])

# essentially probs_grp is a [num_ex, 5, 4] array of probabilities. The 5 'groups' are
# [answer, rationale_conditioned_on_a0, rationale_conditioned_on_a1,
#          rationale_conditioned_on_a2, rationale_conditioned_on_a3].
# We will flatten this to a CSV file so it's easy to submit.
group_names = ["answer"] + [f"rationale_conditioned_on_a{i}" for i in range(4)]
probs_df = pd.DataFrame(
    data=probs_grp.reshape((-1, 20)),
    columns=[f"{group_name}_{i}" for group_name in group_names for i in range(4)],
)

probs_df["annot_id"] = ids_grp

probs_df = probs_df.set_index("annot_id", drop=True)
probs_df.to_csv(output_path)
