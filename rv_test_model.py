# specific to kitti dataset ...
import wrapper
import tensorflow as tf
import numpy as np
import cv2
import os
import pykitti
import csv
from scipy.misc import imresize

from rv_utils import rv_imgCropCenter, create_local_row_kitti

print("--> tensorflow version: " + tf.__version__)
print("--> opencv version: " + cv2.__version__)

basedir = '/media/radu/data/datasets/Kitti'
targetDir = '/media/radu/data/python/bdd_driving/results/kitti'

# Specify the dataset to load
date = '2011_09_26'
drive = '0029'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
cvfont = cv2.FONT_HERSHEY_SIMPLEX
cvfont_size = 0.5
cvfont_thickness = 1
batch_size = 32
win_size = 20
IMSZ = 228

noFrames = dataset.__len__()
iter_cam2 = iter(dataset.cam2)
iter_oxts = iter(dataset.oxts)

model_input_w = 640
model_input_h = 360

a = wrapper.Wrapper("discrete_fcn_lstm", 
            "/media/radu/data/python/bdd_driving/data/discrete_fcn_lstm/model.ckpt-315001.bestmodel", win_size, batch_size)

print("found {:04d} files".format(noFrames))

action_map = {-1:'not_sure', 0:'straight', 1:'slow_or_stop',
                        2:'turn_left', 3:'turn_right',
                        4:'turn_left_slight', 5:'turn_right_slight'}


fid = open(targetDir + "/" + date + "_drive_" + drive + "_sync_full.csv", 'wb')
writer = csv.writer(fid, delimiter=',')

csv_header = ('img_idx', 'vf (m/s)', 'af (m/s^2)', 'wz (deg/s)', 'res0', 'res1', 'res2', 'res3', 'res4', 'res5')
writer.writerow(csv_header)

batch_img = [] # batch_size x win_size x H x W x 3
win_img = win_size * [np.zeros((IMSZ, IMSZ, 3), dtype=np.uint8)] # win_zise x H x W x 3
batch_oxts = []
batch_idx = 0
local_k = []

#for k in range(noFrames):

for k in range(0, noFrames): # read every 3 frames, since in Kitti we have 10 fps
	#print("--> processing img: {:05d}/{:05d}".format(k, noFrames))
	img = np.array(next(iter_cam2))
	img_crop = imresize(rv_imgCropCenter(img, model_input_w, model_input_h), (IMSZ, IMSZ))
	local_oxts = next(iter_oxts)
	
	win_img.append(img_crop)
	if len(win_img) > win_size:
		win_img = win_img[-win_size:]
		batch_img.append(np.stack(win_img))
		batch_oxts.append(local_oxts)
		local_k.append(k)
	
	if len(batch_img) == batch_size:
		res_aux = a.observe_batch(np.stack(batch_img))
		res_aux = np.reshape(res_aux, (batch_size, win_size, res_aux.shape[1]))
		res = res_aux[:, -1, :]
		for idx in range(batch_size):
			local_row = create_local_row_kitti(local_k[idx- batch_size], batch_oxts[idx], res[idx, :])
			print("----> processing batch: {:05d}/{:05f}".format(batch_idx, np.floor(noFrames/batch_size)) + \
				" ==> result: ", local_row)
			writer.writerow(local_row)
		batch_img = []
		batch_oxts = []
		local_k = []
		batch_idx = batch_idx+ 1
	#dummy_frame = next(iter_cam2)
	#dummy_frame = next(iter_cam2)

if len(local_k) > 0:
	for j in range(batch_size - len(local_k)):
		batch_img.append(np.zeros((win_size, IMSZ, IMSZ, 3), dtype=np.uint8))
	res_aux = a.observe_batch(np.stack(batch_img))
	res_aux = np.reshape(res_aux, (batch_size, win_size, res_aux.shape[1]))
	res = res_aux[:, -1, :]
	for idx in range(len(local_k)):
			local_row = create_local_row_kitti(local_k[idx], batch_oxts[idx], res[idx, :])
			print("----> processing batch: {:05d}/{:05f}".format(batch_idx, np.floor(noFrames/batch_size)) + \
				" ==> result: ", local_row)
			writer.writerow(local_row)

fid.close()

