import numpy as np


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
