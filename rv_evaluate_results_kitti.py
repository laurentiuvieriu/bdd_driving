# specific to kitti dataset ...
## we compute the discrete labels on Kitti, based on the instructions provided in: 
# @article{xu2016end,
#   title={End-to-end learning of driving models from large-scale video datasets},
#   author={Xu, Huazhe and Gao, Yang and Yu, Fisher and Darrell, Trevor},
#   journal={arXiv preprint arXiv:1612.01079},
#   year={2016}
# }

# "We first consider the discrete action case, in which we define four actions: straight, stop, left turn, right turn. \
# The task is defined as predicting the feasible actions in the next 1/3 of a second. Specifically, we have as ground \
# truth the vehicle’s speed and its angular velocity between the cur- rent frame and the frame immediately following. \
# We define the action turning right as the event of an angular speed larger than 1.0◦/s and turning left as an angular \
# speed less than −1.0◦/s. Otherwise, if the vehicle’s speed is less than 2.0m/s or the acceleration is less than −1.0m/s2, \
# we de- fine the action stop or slow. The stop or slow action aims to describe when the car has to act in order to avoid, \
# for instance, a crash or a violation of traffic rules. In all other cases, the car’s action is defined as go straight."

## here, we compute a frame discrete label, giving priority (as suggested) to turning actions (left, right), then addressing \
# the remaining cases for fixing "straight" and "slow or stop" labels

import numpy as np
import csv
from sklearn.metrics import accuracy_score


targetDir = '/media/radu/data/python/bdd_driving/results/kitti'
# Specify the dataset to load
date = '2011_09_26'
drive = '0029'

fid = open(targetDir + "/" + date + "_drive_" + drive + "_sync_full.csv", "rb")
reader = csv.reader(fid, delimiter=',')

# csv_header = ('img_idx', 'vf (m/s)', 'af (m/s^2)', 'wz (deg/s)', 'res0', 'res1', 'res2', 'res3', 'res4', 'res5')
# action_map = {-1:'not_sure', 0:'straight', 1:'slow_or_stop',
#                         2:'turn_left', 3:'turn_right',
#                         4:'turn_left_slight', 5:'turn_right_slight'}

# action mapping:
#    straight.......... 0
#    slow or stop...... 1
#    turn_left......... 2
#    turn_right........ 3
#    turn_left_slight.. 2
#    turn_right_slight..3

csv_header = next(reader)
action_gt = []
action_pred = []
for row in reader:
    local_vals = [float(item) for item in row]
    if local_vals[3] > 1.0:
        action_gt.append(3)
    elif local_vals[3] < -1.0:
        action_gt.append(2)
    else:
        if (local_vals[1] < 2.0) or (local_vals[2] < -1.0):
            action_gt.append(1)
        else:
            action_gt.append(0)
    amax_pred = np.argmax(local_vals[-5:])
    if amax_pred >= 4:
        action_pred.append(amax_pred - 2)
    else:
        action_pred.append(amax_pred)

action_gt_fin = np.asarray(action_gt[1:])
action_pred_fin = np.asarray(action_pred[0:-1])

print("--> accuracy on {1}_drive{2}: {:3.3f}".format(date, drive, accuracy_score(action_gt_fin, action_pred_fin)))



