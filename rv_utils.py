import numpy as np
import wrapper
import csv
import pykitti
from scipy.misc import imresize

def rv_imgCropCenter(img, w, h):
    img_size = img.shape
    w_start = 0
    w_stop = img_size[1]
    h_start = 0
    h_stop = img_size[0]
    if w < w_stop:
        w_diff = w_stop - w
        w_start = int(np.floor(w_diff/2))
        w_stop = int(w_start + w)
    if h < h_stop:
        h_diff = h_stop - h
        h_start = int(np.floor(h_diff/2))
        h_stop = int(h_start + h)
    return img[h_start:h_stop, w_start:w_stop, :]

# csv_header = ('img_idx', 'vf (m/s)', 'af (m/s^2)', 'wz (deg/s)', 'res0', 'res1', 'res2', 'res3', 'res4', 'res5')
# input args: local_oxts[0].vf[.af, .wz], res[0:5]

def create_local_row_kitti(local_k, local_oxts, res):
    local_row = ("{:05d}".format(local_k), \
                 "{:03.3f}".format(local_oxts[0].vf), \
                 "{:03.3f}".format(local_oxts[0].af), \
                 "{:03.3f}".format((local_oxts[0].wz)*180/np.pi), \
                 "{:03.3f}".format(res[0]), \
                 "{:03.3f}".format(res[1]), \
                 "{:03.3f}".format(res[2]), \
                 "{:03.3f}".format(res[3]), \
                 "{:03.3f}".format(res[4]), \
                 "{:03.3f}".format(res[5]))
    return local_row

class rv_fcn_lstm_kitti:
    def __init__(self, baseDir, projDir, batchSize = 32, winSize = 20, imSize = 228, modelInputW = 640, modelInputH = 360):
        self.baseDir = baseDir
        self.targetDir = projDir + "/results/kitti"
        self.batchSize = batchSize
        self.winSize = winSize
        self.imSize = imSize
        self.modelInW = modelInputW
        self.modelInH = modelInputH
        
        self.wrapper = wrapper.Wrapper("discrete_fcn_lstm", 
                projDir + "/data/discrete_fcn_lstm/model.ckpt-315001.bestmodel", self.winSize, self.batchSize)
        
    def process_KittiSequence(self, date, drive):
        
        dataset = pykitti.raw(self.baseDir, date, drive, imformat='cv2')
        noFrames = dataset.__len__()
        iter_cam2 = iter(dataset.cam2)
        iter_oxts = iter(dataset.oxts)
        print("--> processing: {:}_drive{:} - found {:04d} files".format(date, drive, noFrames))
        
        fid = open(self.targetDir + "/" + date + "_drive_" + drive + "_sync_full.csv", 'wb')
        writer = csv.writer(fid, delimiter=',')
        
        csv_header = ('img_idx', 'vf (m/s)', 'af (m/s^2)', 'wz (deg/s)', 'res0', 'res1', 'res2', 'res3', 'res4', 'res5')
        writer.writerow(csv_header)
        
        batch_img = [] # batch_size x win_size x H x W x 3
        win_img = self.winSize * [np.zeros((self.imSize, self.imSize, 3), dtype=np.uint8)] # win_zise x H x W x 3
        batch_oxts = []
        batch_idx = 0
        local_k = []
        
        for k in range(0, noFrames): # read every 3 frames, since in Kitti we have 10 fps
            #print("--> processing img: {:05d}/{:05d}".format(k, noFrames))
            img = np.array(next(iter_cam2))
            img_crop = imresize(rv_imgCropCenter(img, self.modelInW, self.modelInH), (self.imSize, self.imSize))
            local_oxts = next(iter_oxts)
            
            win_img.append(img_crop)
            if len(win_img) > self.winSize:
                win_img = win_img[-self.winSize:]
                batch_img.append(np.stack(win_img))
                batch_oxts.append(local_oxts)
                local_k.append(k)
            
            if len(batch_img) == self.batchSize:
                res_aux = self.wrapper.observe_batch(np.stack(batch_img))
                res_aux = np.reshape(res_aux, (self.batchSize, self.winSize, res_aux.shape[1]))
                res = res_aux[:, -1, :]
                for idx in range(self.batchSize):
                    local_row = create_local_row_kitti(local_k[idx- self.batchSize], batch_oxts[idx], res[idx, :])
                    print("----> processing batch: {:05d}/{:05f}".format(batch_idx, np.floor(noFrames/self.batchSize)) + \
                        " ==> result: ", local_row)
                    writer.writerow(local_row)
                batch_img = []
                batch_oxts = []
                local_k = []
                batch_idx = batch_idx+ 1
            #dummy_frame = next(iter_cam2)
            #dummy_frame = next(iter_cam2)
        
        if len(local_k) > 0:
            for j in range(self.batchSize - len(local_k)):
                batch_img.append(np.zeros((self.winSize, self.imSize, self.imSize, 3), dtype=np.uint8))
            res_aux = self.wrapper.observe_batch(np.stack(batch_img))
            res_aux = np.reshape(res_aux, (self.batchSize, self.winSize, res_aux.shape[1]))
            res = res_aux[:, -1, :]
            for idx in range(len(local_k)):
                    local_row = create_local_row_kitti(local_k[idx], batch_oxts[idx], res[idx, :])
                    print("----> processing batch: {:05d}/{:05f}".format(batch_idx, np.floor(noFrames/self.batchSize)) + \
                        " ==> result: ", local_row)
                    writer.writerow(local_row)
        
        fid.close()
