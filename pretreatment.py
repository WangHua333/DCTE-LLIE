import numpy as np
import cv2
import main
from glob import glob
import os
import tensorflow as tf
from PIL import Image

# 此代码用于生成map, 没有在main和model中调用, 可做改进调试之用
def save_images(images, file_path):
    """
    将 NumPy 数组保存为图像文件

    Args:
        images: 要保存的图像数据，NumPy 数组，shape 为 (H, W, C) 或 (N, H, W, C)，取值范围为 [0, 1] 或 [0, 255]。
        file_path: 保存的文件路径，包括文件名和后缀，如 'example.png'。

    Returns:
        无返回值。
    """
    # 转换数据类型和取值范围
    if images.dtype == np.float32 or images.dtype == np.float64:
        images = np.uint8(np.clip(images, 0, 1) * 255.0)
    elif images.dtype == np.uint8:
        images = np.array(images)
    else:
        raise ValueError('Unsupported data type: %s' % images.dtype)

    # 处理多张图片的情况
    if len(images.shape) == 4:
        N, H, W, C = images.shape
        for i in range(N):
            image_i = images[i, :, :, :]
            image_i = cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR
            cv2.imwrite(file_path + '.%d.png' % i, image_i)
        return

    # 处理单张图片的情况
    H, W, C = images.shape
    images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)  # 将 RGB 转换为 BGR
    cv2.imwrite(file_path, images)

def get_color_freqs(input_high, channel=3):
    def get_color_freq(im):
        im_shape = tf.shape(im)
        im = tf.reshape(im, [-1, channel])
        im = im * 255.0  # 输入的input_high是0-1值
        k = [1, 1000, 1000000]
        if channel == 4:
            k = [1, 1000, 1000000, 1000000000]
        im = im * k
        sum_im = tf.squeeze(tf.reduce_sum(im, axis=1))
        color_sum, indices = tf.unique(sum_im)
        color_counts = tf.bincount(indices)
        color_freq = tf.gather(color_counts, indices)
        color_freq = tf.reshape(color_freq, [im_shape[0], im_shape[1], 1])
        return tf.cast(color_freq, dtype=tf.float32)

    color_freqs = tf.map_fn(get_color_freq, input_high, dtype=tf.float32)
    return color_freqs


def get_color_Fs(images):
    def get_color_F(im):
        N = tf.reduce_prod(tf.shape(im))
        low_bound = 0.00005 * tf.cast(N, tf.float32)
        high_bound = 0.005 * tf.cast(N, tf.float32)
        F = tf.where(tf.logical_and(im > low_bound, im < high_bound),
                     tf.ones_like(im), tf.zeros_like(im))
        return F

    color_Fs = tf.map_fn(get_color_F, images)
    return color_Fs

def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

high_data_dir = main.args.high_data
attention_map_dir = main.args.attentionMap

# 确保 attention map 和 attention point 目录存在
os.makedirs(attention_map_dir, exist_ok=True)

# 获取待处理文件夹中的所有图片文件
image_files = glob(os.path.join(high_data_dir) + '/*.png')
print('[*] Number of pretraining data: %d' % len(image_files))
image_files.sort()
preImages = []
for idx in range(len(image_files)):
    im = load_images(image_files[idx])
    preImages.append(im)
preImages = np.array(preImages)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    images = tf.placeholder(tf.float32, [None, None, None, 3], name='images')
    freq = get_color_freqs(images)
    F = get_color_Fs(freq)
    attention_map = images * F

    [attention_map] = sess.run([attention_map], feed_dict={images: preImages})
    for idx in range(len(image_files)):
        attention_map_file = os.path.join(attention_map_dir, os.path.basename(image_files[idx]))
        save_images(attention_map[idx], attention_map_file)
    print('[*] Attention map saved to %s' % attention_map_dir)


