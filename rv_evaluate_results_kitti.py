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

import os
import numpy as np
import csv
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

resultsDir = '/data/radu/python/bdd_driving/results/kitti/fcn_lstm'
# Specify the dataset to load
# date = '2011_09_26'

files = [f for f in os.listdir(resultsDir) if f.endswith('full.csv')]
print("--> found {:d} results files".format(len(files)))

all_drives = [item[17:21] for item in files]
all_dates = [item[0:10] for item in files]

def get_labelsFromFile(date, drive, wz_th=1):
	fid = open(resultsDir + "/" + date + "_drive_" + drive + "_sync_full.csv", "rb")
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
		if local_vals[3] > wz_th:
		    action_gt.append(2)
		elif local_vals[3] < -wz_th:
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
	return [action_gt_fin, action_pred_fin]


acc_seq_grid = []
acc_frame_grid = []
cm_frame_grid = []

for wz_th in range(1,21,1):
	all_acc = []
	all_labs_gt = []
	all_labs_pred = []
	for date, drive in zip(all_dates, all_drives):
		labs = get_labelsFromFile(date, drive, wz_th)
		all_labs_gt.append(labs[0])
		all_labs_pred.append(labs[1])
		all_acc.append(accuracy_score(labs[0], labs[1]))
		#print("--> accuracy on {:}_drive{:}: {:3.3f}".format(date, drive, accuracy_score(labs[0], labs[1])))
	print("---> angle th: {:02d} - avg seq-level accuracy: {:.2f}".format(wz_th, np.mean(np.stack(all_acc))))
	print("---> angle th: {:02d} - avg frame-level accuracy: {:.2f}".format(wz_th, accuracy_score(np.hstack(all_labs_gt), np.hstack(all_labs_pred))))
	acc_seq_grid.append(np.mean(np.stack(all_acc)))
	acc_frame_grid.append(accuracy_score(np.hstack(all_labs_gt), np.hstack(all_labs_pred)))
	cm_frame_grid.append(confusion_matrix(np.hstack(all_labs_gt), np.hstack(all_labs_pred)))
	cm_frame_grid[-1] = cm_frame_grid[-1].astype('float') / cm_frame_grid[-1].sum(axis=1)[:, np.newaxis]
	#cm = confusion_matrix(labs[0], labs[1])
	#m = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	#print("--> confusion matrix...")
	#print(cm)

x = range(1,21,1)

plt.plot(x, np.hstack(acc_seq_grid), color="red", linewidth=2.0, label="acc_sequence_based")
plt.plot(x, np.hstack(acc_frame_grid), color="blue", linewidth=2.0, label="acc_frame_based")

plt.grid(True)

plt.xlabel('rot. acceleration threshold')
plt.ylabel('4 class accuracy')
plt.title('fcn_lstm accuracy on kitti')

plt.legend(loc='upper left')
plt.show()

#np.set

#cm = confusion_matrix(labs[0], labs[1])
#cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print("--> confusion matrix...")
#print(cm)





