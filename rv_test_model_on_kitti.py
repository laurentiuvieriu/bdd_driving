import wrapper
import tensorflow as tf
from tensorflow.core.example import example_pb2
from cStringIO import StringIO
from PIL import Image
from matplotlib.pyplot import imshow
%matplotlib inline
import numpy as np

a = wrapper.Wrapper("discrete_tcnn1", 
            "/data/yang/code/BDD_Driving_Model/data/discrete_tcnn1/model.ckpt-126001.bestmodel",
            20)

