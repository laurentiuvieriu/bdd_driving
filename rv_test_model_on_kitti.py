## unzip kitti one archive at a time, compute the predictions and delete the images after that
## this approach should limit the space requirements as much as possible
## csv files with results can be found in <projDir>/kitti/fcn_lstm

import os
from rv_utils import rv_fcn_lstm_kitti
datasetDir = '/media/radu/data/datasets/Kitti'
projDir = '/media/radu/data/python/bdd_driving'

fileList = [f for f in os.listdir(datasetDir) if f.endswith('sync.zip')]
print("--> looking for zip files in {:} - found: {:03} files".format(datasetDir, len(fileList)))

model = rv_fcn_lstm_kitti(datasetDir, projDir)

for k in range(len(fileList)):
	comm = []
	comm.append("unzip -oq {:}/{:} -d {:}/".format(datasetDir, fileList[k], datasetDir))
	print("--> executing: {:}".format(comm[-1]))
	os.system(comm[-1])
	print("--> done... now running the model ...")

	date = fileList[k][0:10]
	drive = fileList[k][17:21]

	model.process_KittiSequence(date, drive)

	comm.append("rm -rf {:}/{:}/{:}".format(datasetDir, date, fileList[k][0:-4]))
	print("--> executing: {:}".format(comm[-1]))
	os.system(comm[-1])
	print("--> done")

# comm.append("/home/radu/local/tensorflow_011/bin/python rv_evaluate_results_kitti.py")
# print("--> executing: {:}".format(comm[-1]))
# os.system(comm[-1])
# print("--> done")


