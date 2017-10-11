# specific to kitti dataset ...
import numpy as np
import cv2
import os
import pykitti
import csv

from rv_utils import rv_imgCropCenter, create_local_row_kitti

print("--> opencv version: " + cv2.__version__)

basedir = '/media/radu/sdb_data/radu/work/datasets/Kitti'
targetDir = '/media/radu/sdb_data/radu/work/python/bdd_driving/results/kitti'

# Specify the dataset to load
date = '2011_09_30'
drive = '0028'
dataset = pykitti.raw(basedir, date, drive, imformat='cv2')
cvfont = cv2.FONT_HERSHEY_SIMPLEX
cvfont_size = 0.5
# cvfont_thickness = 1
# batch_size = 32
# win_size = 20
# IMSZ = 228

fid = open(targetDir + "/" + date + "_drive_" + drive + "_sync.csv", "rb")
reader = csv.reader(fid, delimiter=',')
csv_header = next(reader)

line = next(reader)





