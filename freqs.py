import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import argparse

from utils import load_images

# 此代码用于图像颜色频率统计测试
def get_color_freqs(input_high):
    def get_color_freq(im):
        im_shape = tf.shape(im)
        im = tf.reshape(im, [-1, 3])
        im = im * 255.0  # 输入的input_high是0-1值
        im = im * tf.constant([1, 1000.0, 1000000.0])
        sum_im = tf.squeeze(tf.reduce_sum(im, axis=1))
        im_colors = sum_im
        color_sum, indices = tf.unique(sum_im)
        color_counts = tf.bincount(indices)
        color_freq = tf.gather(color_counts, indices)

        color_freq = tf.reshape(color_freq, [im_shape[0], im_shape[1], 1])
        im_colors = tf.reshape(im_colors, [im_shape[0], im_shape[1], 1])
        return tf.cast(color_freq, dtype=tf.float32), tf.cast(im_colors, dtype=tf.float32)

    color_freqs, colors = tf.map_fn(get_color_freq, input_high, dtype=(tf.float32, tf.float32))
    return color_freqs, colors


def get_color_Fs(images):
    def get_color_F(im):
        N = tf.reduce_prod(tf.shape(im))
        low_bound = 0.0000005 * tf.cast(N, tf.float32)
        high_bound = 1 * tf.cast(N, tf.float32)
        F = tf.where(tf.logical_and(im > low_bound, im < high_bound),
                     tf.ones_like(im), tf.zeros_like(im))
        return F

    color_Fs = tf.map_fn(get_color_F, images)
    return color_Fs

# a = np.ones([1, 200, 600, 3])
# b = np.zeros([1, 200, 600, 3])
# t = np.concatenate([a, b], axis=1)
# t = np.concatenate([t, t], axis=0)
parser = argparse.ArgumentParser(description='')
parser.add_argument('--test_low_dir', dest='test_low_dir', default='./data/all500/low',
                    help='directory for testing inputs (low)')
parser.add_argument('--test_high_dir', dest='test_high_dir', default='./data/all500/high',
                    help='directory for testing inputs (high)')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    images = tf.placeholder(tf.float32, [None, None, None, 3], name='images')
    color_freqs, colors = get_color_freqs(images)
    # Fs = get_color_Fs(color_freqs)
    # attention_map = images * Fs

    test_low_data_name = glob(os.path.join(args.test_low_dir) + '/*.*')
    test_high_data_name = glob(os.path.join(args.test_high_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_high_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)
        test_high_im = load_images(test_high_data_name[idx])
        test_high_data.append(test_high_im)
    test_low_data = np.array(test_low_data)
    test_high_data = np.array(test_high_data)
    print(test_high_data.shape)
    np.set_printoptions(threshold=np.inf, suppress=True)

    [ims, ims_colors] = sess.run([color_freqs, colors], feed_dict={images: test_high_data})
    # print(amap)
    # print(ims.shape)
    # low_bound = 0.0005
    # high_bound = 0.005
    # every_freqs = []
    # count_list = []
    # for i in range(len(test_high_data_name)):
    #     im = np.reshape(ims[i, ...], [1, -1])
    #     category, indices = np.unique(im, return_index=True)
    #     freq = category / 240000.0
    #     freq = np.sort(freq)
    #     every_freqs.append(freq)
    #     print('freq.shape = ', freq.shape)
    #     print(freq)
    #     low_m = np.ones(freq.shape, dtype=float) * low_bound
    #     high_m = np.ones(freq.shape, dtype=float) * high_bound
    #     min_count = len(np.where(freq < low_m)[0])
    #     mid_count = len(np.where((low_m <= freq) * (freq <= high_m))[0])
    #     max_count = len(np.where(freq > high_m)[0])
    #     count_list.append(np.array([min_count, mid_count, max_count]))
    #
    #     y1 = freq[np.where(freq < low_m)]
    #     x1 = np.arange(min_count)
    #     y2 = freq[np.where((low_m <= freq) * (freq <= high_m))]
    #     x2 = np.arange(min_count, min_count + mid_count)
    #     y3 = freq[np.where(freq > high_m)]
    #     x3 = np.arange(min_count + mid_count, min_count + mid_count + max_count)
    #     plt.title('image %d' % i)
    #     plt.plot(x1, y1, x2, y2, x3, y3,)
    #     plt.show()
    # print(count_list)
    # count_np = np.array(count_list)
    # avg = np.average(count_np, axis=0)
    # print(count_np)
    # print(avg)
    # print(len(every_freqs))

    # for i in range(len(test_high_data_name)):
    #     im = np.reshape(ims[i, ...], [1, -1])
    low_bound = 0.00005 * 240000.0
    high_bound = 0.005 * 240000.0
    every_freqs = []
    count_list = []
    ims = np.concatenate([ims, ims_colors], axis=-1)
    print('ims:', ims.shape)
    for i in range(500):
        print('image: ', i)
        im = ims[i]
        im = np.reshape(im, [-1, 2])
        category, indices = np.unique(im, return_inverse=True, axis=0)

        freqs = np.sort(category[:, 0])
        print('freqs.shape: ', freqs.shape)
        every_freqs.append(freqs)

        low_m = np.ones(freqs.shape, dtype=float) * low_bound
        high_m = np.ones(freqs.shape, dtype=float) * high_bound
        min_count = len(np.where(freqs < low_m)[0])
        mid_count = len(np.where((low_m <= freqs) * (freqs <= high_m))[0])
        max_count = len(np.where(freqs > high_m)[0])
        counts = np.array([min_count, mid_count, max_count])
        print('counts: ', counts)
        count_list.append(counts)

        y1 = freqs[np.where(freqs < low_m)]
        x1 = np.arange(min_count)
        y2 = freqs[np.where((low_m <= freqs) * (freqs <= high_m))]
        x2 = np.arange(min_count, min_count + mid_count)
        y3 = freqs[np.where(freqs > high_m)]
        x3 = np.arange(min_count + mid_count, min_count + mid_count + max_count)
        plt.title('image %d' % i)
        plt.plot(x1, y1, x2, y2, x3, y3)
        plt.show()




