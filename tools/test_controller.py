# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import vilbert.utils as utils


controller = utils.MultiTaskStopOnPlateau(
    mode="max", patience=1, continue_threshold=0.01, cooldown=1, threshold=0.001
)


"""
=========
test case
=========


"""
scores = [10, 20, 30, 40, 50, 50, 50, 50, 49, 48, 50, 51, 52, 52, 52, 52, 52]

for s in scores:
    controller.step(s)
    print(s, controller.in_stop)
