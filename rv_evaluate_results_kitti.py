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
# truth the vehicles speed and its angular velocity between the current frame and the frame immediately following. \
# We define the action turning right as the event of an angular speed larger than 1.0deg/s and turning left as an angular \
# speed less than -1.0deg/s. Otherwise, if the vehicles speed is less than 2.0m/s or the acceleration is less than -1.0m/s2, \
# we define the action stop or slow. In all other cases -> go straight

## here, we compute a frame discrete label, giving priority (as suggested) to turning actions (left, right), then addressing \
# the remaining cases for fixing "straight" and "slow or stop" labels

import numpy as np
import csv
from sklearn.metrics import accuracy_score, confusion_matrix


basedir = '/media/radu/sdb_data/radu/work/datasets/Kitti'
targetDir = '/media/radu/sdb_data/radu/work/python/bdd_driving/results/kitti'
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
        action_gt.append(2)
    elif local_vals[3] < -1.0:
        action_gt.append(3)
    else:
        if (local_vals[1] < 2.0) or (local_vals[2] < -1.0):
            action_gt.append(1)
        else:
            action_gt.append(0)
    amax_pred = np.argmax(local_vals[-6:])
    if amax_pred >= 4:
        action_pred.append(amax_pred - 2)
    else:
        action_pred.append(amax_pred)

action_gt_fin = np.asarray(action_gt[1:])
action_pred_fin = np.asarray(action_pred[0:-1])

print("--> accuracy on {:}_drive{:}: {:3.3f}".format(date, drive, accuracy_score(action_gt_fin, action_pred_fin)))
cm = confusion_matrix(action_gt_fin, action_pred_fin)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("--> confusion matrix...")
print(cm)



