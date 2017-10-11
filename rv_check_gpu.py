'''
This is a wrapper for easier evaluation.

The original evaluation code is mainly written for evaluation on a validation set. It requires the dataset be processed
into the designated TFRecord format. This wrapper aims to be a lightweight interface without requiring the TFRecord
format input. Instead, it accepts inputs of images, and output actions on the fly.
'''

import tensorflow as tf
import models.car_stop_model as model
from scipy.misc import imresize
import numpy as np
import json

# The following import populates some FLAGS default value
import data_providers.nexar_large_speed
import batching
import dataset
import util_car
import csv

def write_dict_to_file(filename, mydict):
	with open(filename, 'wb') as csv_file:
		writer = csv.writer(csv_file)
		for key, value in mydict.items():
		   writer.writerow([key, value])

FLAGS = tf.app.flags.FLAGS
flags_passthrough = FLAGS._parse_flags()
from config import common_config, common_config_post

IMSZ = 228
model_config_name = "discrete_fcn_lstm"
truncate_len = 20

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#write_dict_to_file('./results/flags_orig.txt', FLAGS.__flags)

import config
common_config("eval")
config_fun = getattr(config, model_config_name)
config_fun("eval")
common_config_post("eval")

#write_dict_to_file('./results/flags_altered.txt', FLAGS.__flags)

# Tensors in has the format: [images, speed] for basic usage, excluding only_seg
# For now, we decide not to support previous speed as input, thus we use a fake speed (-1) now
# and ensures the speed is not used by asserting FLAGS.use_previous_speed_feature==False
assert (not hasattr(FLAGS, "use_previous_speed_feature")) or (FLAGS.use_previous_speed_feature == False)
# batch size 1 all the time, length undetermined, width and height are IMSZ
#FLAGS.__setattr__('log_device_placement', True)
tensors_in = tf.placeholder(tf.uint8, shape=(256, truncate_len, IMSZ, IMSZ, 3), name="images_input")
speed = None

logits_all = model.inference([tensors_in, speed], -1, for_training=False)

# Restore the moving average version of the learned variables for eval.
variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)

config = tf.ConfigProto(intra_op_parallelism_threads=1)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
