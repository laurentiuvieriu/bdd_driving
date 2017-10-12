# specific to kitti dataset ...
import numpy as np
import cv2
import os
import pykitti
import csv

from rv_utils import rv_imgCropCenter, create_local_row_kitti

print("--> opencv version: " + cv2.__version__)

basedir = '/media/radu/data/datasets/Kitti'
targetDir = '/media/radu/data/python/bdd_driving/results/kitti'

# Specify the dataset to load
date = '2011_09_26'
drive = '0029'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
font = cv2.FONT_HERSHEY_SIMPLEX
font_size = 0.5
font_thickness = 1
rect_width = 5
rect_max = 30 # maximum length of a line in pixels
rect_center_x = 240
model_input_w = 640
model_input_h = 360

noFrames = dataset.__len__()

fid = open(targetDir + "/" + date + "_drive_" + drive + "_sync_full.csv", "rb")
reader = csv.reader(fid, delimiter=',')
# row_count = np.sum(1 for row in reader)
csv_header = next(reader)

all_labs = []
for k in range(noFrames):
    line = next(reader)
    all_labs.append(np.asarray([float(item) for item in line]))

all_labs = np.stack(all_labs)
fid.close()
lab_max = all_labs.max(0)
lab_min = all_labs.min(0)
max_stretch = []
max_stretch.append(np.abs(lab_max))
max_stretch.append(np.abs(lab_min))
max_stretch = np.stack(max_stretch)
lab_range = max_stretch.max(0)
lab_range[-6:]= 1

iter_cam2 = iter(dataset.cam2)

def plot_bar_on_img(img, mag, x_center, y_pos, bar_width, label, font, font_size, font_thickness):
    cv2.putText(img_crop, label, (10, y_pos), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    if mag < 0:
        cv2.rectangle(img, (x_center+ mag, y_pos- bar_width), (x_center, y_pos), (0,0,255), bar_width)
    else:
        cv2.rectangle(img, (x_center, y_pos- bar_width), (x_center+ mag, y_pos), (0,255,0), bar_width)
    return img

# assume the results and the img files are synchronized
# csv_header = ('img_idx', 'vf (m/s)', 'af (m/s^2)', 'wz (deg/s)', 'res0', 'res1', 'res2', 'res3', 'res4', 'res5')
# action_map = {-1:'not_sure', 0:'straight', 1:'slow_or_stop',
#                         2:'turn_left', 3:'turn_right',
#                         4:'turn_left_slight', 5:'turn_right_slight'}

for k in range(noFrames):
    row_pos = iter(range(20, 360, 20))
    img = np.array(next(iter_cam2))
    img_crop = rv_imgCropCenter(img, model_input_w, model_input_h)
    local_res = all_labs[k]
    labels = [float(item) for item in local_res]
    cv2.putText(img_crop, "velocity fwd (vf)...................... {:02.2f}".format(local_res[1]), (10,next(row_pos)), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    # deal with acceleration
    accel_mag_gt = int(np.round(local_res[2]/lab_range[2]*rect_max))
    y_pos = next(row_pos)
    plot_bar_on_img(img_crop, accel_mag_gt, rect_center_x, y_pos, rect_width, 'accel_gt:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[2]), (280, y_pos), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    row_pos_local = next(row_pos)
    fwd_pred = int(np.round(local_res[4]/lab_range[4]*rect_max))
    plot_bar_on_img(img_crop, fwd_pred, rect_center_x, row_pos_local, rect_width, 'fwd/slow_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[4]), (320, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    slow_pred = -int(np.round(local_res[5]/lab_range[5]*rect_max))
    plot_bar_on_img(img_crop, slow_pred, rect_center_x, row_pos_local, rect_width, 'fwd/slow_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[5]), (280, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    # deal with steering
    lr_mag_gt = -int(np.round(local_res[3]/lab_range[3]*rect_max))
    y_pos = next(row_pos)
    plot_bar_on_img(img_crop, lr_mag_gt, rect_center_x, y_pos, rect_width, 'left/right_gt:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[3]), (280, y_pos), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    row_pos_local = next(row_pos)
    left_pred = -int(np.round(local_res[6]/lab_range[6]*rect_max))
    plot_bar_on_img(img_crop, left_pred, rect_center_x, row_pos_local, rect_width, 'left/right_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[6]), (280, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    right_pred = int(np.round(local_res[7]/lab_range[7]*rect_max))
    plot_bar_on_img(img_crop, right_pred, rect_center_x, row_pos_local, rect_width, 'left/right_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[7]), (320, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    # deal with slight_steering    
    row_pos_local = next(row_pos)
    slight_left_pred = -int(np.round(local_res[8]/lab_range[8]*rect_max))
    plot_bar_on_img(img_crop, slight_left_pred, rect_center_x, row_pos_local, rect_width, 'slight left/right_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[8]), (280, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    slight_right_pred = int(np.round(local_res[9]/lab_range[9]*rect_max))
    plot_bar_on_img(img_crop, slight_right_pred, rect_center_x, row_pos_local, rect_width, 'slight left/right_pred:', font, font_size, font_thickness)
    cv2.putText(img_crop, "{:.2f}".format(local_res[9]), (320, row_pos_local), font, font_size, (0,255,0), font_thickness, cv2.LINE_AA)
    
    cv2.imshow('image', img_crop)
    cv2.moveWindow('image', 2300, 360)
    cv2.waitKey(30)
    #cv2.imwrite(targetDir+"/" + date + "_drive_" + drive + "_sync_full/frame_{:010d}.png".format(k), img_crop)

# line = next(reader)





