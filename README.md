# bdd_driving

BDD driving project tested on Kitti

Kitti raw sequences downloaded from: http://www.cvlibs.net/datasets/kitti/raw_data.php?type=city

Distribution of scenes (the ones considered here):
    City ............... 28
    Residential ........ 21
    Road ............... 12
================================
    total .............. 61
    
Total number of processed frames (in all considered scenes): 42321

Details of the experiment:
    - approach from https://arxiv.org/pdf/1612.01079.pdf (check v1 as well for additional details)
    - code: https://github.com/gy20073/BDD_Driving_Model
    - model used: discrete_fcn_lstm/model.ckpt-315001.bestmodel
    - the model predicts one of the 6 discrete labels:
        - straight.......... 0
    	- slow or stop...... 1
    	- turn_left......... 2
    	- turn_right........ 3
    	- turn_left_slight.. 2
    	- turn_right_slight..3
    - the mapping here colapses the last two classes into the corresponding "turn_left"/"turn_right" ones, to be inline with the original paper
    - from Kitti, we use forward acceleration (m/s^2), velocity (m/s) and yaw acceleration (deg/s) to define the same 4 classes, as described in the v1 of the paper
    - the model takes in a sequence of (consecutive) frames and predicts one of the 6 classes (the mapping to 4 classes is performed afterwards)
    - a length of 20 frames (2 seconds worth) per sequence has been used in this experiment (first sequences have been 0-padded)
    - the frames from Kitti are wide frames, 1242 x 375 pixels, whereas the model expects frames of 640 x 360 frames reshaped into 228 x 228, therefore central cropping has been performed
