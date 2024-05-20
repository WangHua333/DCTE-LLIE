from glob import glob

import numpy as np
import tensorflow as tf

from utils import load_images

high_image_names = glob('./data/all500/high/*.png')
high_image_names.sort()
high_image_list = []
for idx in range(len(high_image_names)):
    high_im = load_images(high_image_names[idx])
    high_image_list.append(high_im)

high_image_np = np.array(high_image_list)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    high_data = tf.placeholder(tf.float32, [None, None, None, 3], name='high_data')




