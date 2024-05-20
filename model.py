from __future__ import print_function
import os
import time
import random

from PIL import Image
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from utils import *
from skimage import color


def psnr(input_high, output):
    input_high = tf.image.convert_image_dtype(input_high, tf.float32)
    output = tf.image.convert_image_dtype(output, tf.float32)
    psnrVal = tf.image.psnr(input_high, output, max_val=1.0)
    ret = psnrVal.eval()[0]
    return ret


def ssim(input_high, output):
    input_high = tf.image.convert_image_dtype(input_high, tf.float32)
    output = tf.image.convert_image_dtype(output, tf.float32)
    ssimVal = tf.image.ssim(input_high, output, max_val=1.0)
    ret = ssimVal.eval()[0]
    return ret


def lab2rgb(lab_images):
    lab_shape = tf.shape(lab_images)
    rgb_images = tf.py_func(color.lab2rgb, [lab_images], tf.float32)
    # rgb_images = tf.clip_by_value(rgb_images, 0, 1)
    rgb_images = tf.reshape(rgb_images, [lab_shape[0], lab_shape[1], lab_shape[2], 3])
    return rgb_images


def rgb2lab(rgb_images):
    # rgb_images 取值0-1
    rgb_shape = tf.shape(rgb_images)
    lab_images = tf.py_func(color.rgb2lab, [rgb_images], tf.float32)
    lab_images = tf.reshape(lab_images, [rgb_shape[0], rgb_shape[1], rgb_shape[2], 3])
    return lab_images


# def lab2rgb(lab_image):
#     # 将 Lab 转换为 XYZ
#     L, a, b = tf.unstack(lab_image, axis=-1)
#     fY = (L + 16) / 116
#     fX = a / 500 + fY
#     fZ = fY - b / 200
#     X = tf.where(fX > 0.206893, fX ** 3, (fX - 16 / 116) / 7.787)
#     Y = tf.where(fY > 0.206893, fY ** 3, (fY - 16 / 116) / 7.787)
#     Z = tf.where(fZ > 0.206893, fZ ** 3, (fZ - 16 / 116) / 7.787)
#     # 归一化 D65 白点
#     X *= 0.950456
#     Y *= 1.0
#     Z *= 1.088754
#     # 将 XYZ 转换为 RGB
#     rgb_image = tf.matmul(tf.stack([X, Y, Z], axis=-1), tf.constant([
#         [3.24048134, -1.53715152, -0.49853633],
#         [-0.96925495, 1.87599, 0.04155506],
#         [0.05564664, -0.20404134, 1.05731107]
#     ]))
#     # 将 0-1 转换为 0-255
#     # rgb_image = rgb_image * 255.0
#     return rgb_image
#
#
# def rgb2lab(rgb_image):
#     # 将 0-255 转换为 0-1
#     # rgb_image = rgb_image / 255.0
#     # 将 RGB 转换为 XYZ
#     XYZ = tf.matmul(rgb_image, tf.constant([
#         [0.412453, 0.357580, 0.180423],
#         [0.212671, 0.715160, 0.072169],
#         [0.019334, 0.119193, 0.950227]
#     ]))
#     # 归一化 D65 白点
#     X, Y, Z = tf.unstack(XYZ, axis=-1)
#     X /= 0.950456
#     Y /= 1.0
#     Z /= 1.088754
#     # 将 XYZ 转换为 Lab
#     fX = tf.where(X > 0.008856, X ** (1 / 3), 7.787 * X + 16 / 116)
#     fY = tf.where(Y > 0.008856, Y ** (1 / 3), 7.787 * Y + 16 / 116)
#     fZ = tf.where(Z > 0.008856, Z ** (1 / 3), 7.787 * Z + 16 / 116)
#     L = 116 * fY - 16
#     a = 500 * (fX - fY)
#     b = 200 * (fY - fZ)
#     lab_image = tf.stack([L, a, b], axis=-1)
#     return lab_image


def concat(layers):
    return tf.concat(layers, axis=3)


def DecomNet(input_im, layer_num, channel=64, kernel_size=3):
    input_max = tf.reduce_max(input_im, axis=3, keepdims=True)
    input_im = concat([input_max, input_im])
    with tf.variable_scope('DecomNet', reuse=tf.AUTO_REUSE):
        conv = tf.layers.conv2d(input_im, channel, kernel_size * 3, padding='same', activation=None,
                                name="shallow_feature_extraction")
        for idx in range(layer_num):
            conv = tf.layers.conv2d(conv, channel, kernel_size, padding='same', activation=tf.nn.relu,
                                    name='activated_layer_%d' % idx)
        conv = tf.layers.conv2d(conv, 4, kernel_size, padding='same', activation=None, name='recon_layer')

    R = tf.sigmoid(conv[:, :, :, 0:3])
    L = tf.sigmoid(conv[:, :, :, 3:4])

    return R, L


def RelightNet(input_L, input_R, channel=64, kernel_size=3):
    input_im = concat([input_R, input_L])
    with tf.variable_scope('RelightNet'):
        k3n32_1_output = tf.layers.conv2d(input_L, 16, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        # k3n64_k1n128_k3n64_1
        conv = tf.layers.conv2d(k3n32_1_output, 32, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 64, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_1_output = tf.layers.conv2d(conv, 32, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)
        # k3n2
        k3n2_output = tf.layers.conv2d(k3n64_k1n128_k3n64_1_output, 1, kernel_size=3, strides=1, padding='same',
                                       activation=None)
        attention_map = k3n2_output
        input_im = attention_map * input_L
        conv0 = tf.layers.conv2d(input_im, channel, kernel_size, padding='same', activation=None)
        conv1 = tf.layers.conv2d(conv0, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(conv1, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(conv2, channel, kernel_size, strides=2, padding='same', activation=tf.nn.relu)

        up1 = tf.image.resize_nearest_neighbor(conv3, (tf.shape(conv2)[1], tf.shape(conv2)[2]))
        deconv1 = tf.layers.conv2d(up1, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv2
        up2 = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(conv1)[1], tf.shape(conv1)[2]))
        deconv2 = tf.layers.conv2d(up2, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv1
        up3 = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(conv0)[1], tf.shape(conv0)[2]))
        deconv3 = tf.layers.conv2d(up3, channel, kernel_size, padding='same', activation=tf.nn.relu) + conv0

        deconv1_resize = tf.image.resize_nearest_neighbor(deconv1, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        deconv2_resize = tf.image.resize_nearest_neighbor(deconv2, (tf.shape(deconv3)[1], tf.shape(deconv3)[2]))
        feature_gather = concat([deconv1_resize, deconv2_resize, deconv3])
        feature_fusion = tf.layers.conv2d(feature_gather, channel, 1, padding='same', activation=None)
        output = tf.layers.conv2d(feature_fusion, 1, 3, padding='same', activation=None)
    return output


# CWANab
def CWANab(input_R_ab, channel=64, kernel_size=3):
    ab = input_R_ab
    residual = ab
    with tf.variable_scope('CWANab'):
        # k3n32_1
        k3n32_1_output = tf.layers.conv2d(ab, 32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        # k3n64_k1n128_k3n64_1
        conv = tf.layers.conv2d(k3n32_1_output, 64, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 128, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_1_output = tf.layers.conv2d(conv, 64, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)
        # k3n2
        k3n2_output = tf.layers.conv2d(k3n64_k1n128_k3n64_1_output, 2, kernel_size=3, strides=1, padding='same',
                                       activation=None)
        attention_map = k3n2_output
        # 第一部分
        cat_res_att = concat([residual, k3n2_output])
        # k3n32_2
        k3n32_2_output = tf.layers.conv2d(cat_res_att, 32, kernel_size=3, strides=1, padding='same',
                                          activation=tf.nn.relu)
        # k3n64_k1n128_k3n64_2
        conv = tf.layers.conv2d(k3n32_2_output, 64, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 128, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_2_output = tf.layers.conv2d(conv, 64, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)

        # k3n64_k1n128_k3n64_3
        conv = tf.layers.conv2d(k3n64_k1n128_k3n64_2_output, 64, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 128, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_3_output = tf.layers.conv2d(conv, 64, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)

        # k3n4
        k3n4_output = tf.layers.conv2d(k3n64_k1n128_k3n64_3_output, 4, kernel_size=3, strides=1, padding='same',
                                       activation=None)

        attention_points = k3n4_output[:, :, :, 2:]
        cat_output = k3n4_output[:, :, :, :2] + residual
        # 第二部分
        input_ab = attention_map * ab
        k3n32_2_output = tf.layers.conv2d(input_ab, 32, kernel_size=3, strides=1, padding='same',
                                          activation=tf.nn.relu)
        # k3n64_k1n128_k3n64_2
        conv = tf.layers.conv2d(k3n32_2_output, 64, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 128, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_2_output = tf.layers.conv2d(conv, 64, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)

        # k3n64_k1n128_k3n64_3
        conv = tf.layers.conv2d(k3n64_k1n128_k3n64_2_output, 64, kernel_size=3, strides=1, padding='same',
                                activation=tf.nn.relu)
        conv = tf.layers.conv2d(conv, 128, kernel_size=1, strides=1, padding='same',
                                activation=tf.nn.relu)
        k3n64_k1n128_k3n64_3_output = tf.layers.conv2d(conv, 64, kernel_size=3, strides=1, padding='same',
                                                       activation=tf.nn.relu)

        # k3n4
        k3n4_output = tf.layers.conv2d(k3n64_k1n128_k3n64_3_output, 2, kernel_size=3, strides=1, padding='same',
                                       activation=None)
        mul_output = k3n4_output + residual
        # 点乘融合
        # enhance_ab = cat_output * mul_output
        # # 加和
        enhance_ab = cat_output + 0.1 * mul_output
        # # 卷积
        # cat_abab = concat([cat_output, mul_output])
        # cat_aabb = concat([cat_output[:, :, :, 0:1], mul_output[:, :, :, 0:1],
        #                    cat_output[:, :, :, 1:2], mul_output[:, :, :, 1:2]])
        # enhance_ab = tf.layers.conv2d(cat_abab, 2, kernel_size=3, strides=1, padding='same', activation=None)
        # enhance_ab = cat_output
    return enhance_ab, attention_map, attention_points


def get_color_freqs(input_high, channel=3):  # 获取图像的颜色频率图， 输入B * H * W * C
    def get_color_freq(im):  # 输入为H * W * C
        im_shape = tf.shape(im)  # 获取im_shape
        im = tf.reshape(im, [-1, channel])  # 将im转换为H * W行，3列
        im = im * 255.0  # 输入的input_high是0-1值，转化为0-255
        im = im * tf.constant([1.0, 1000.0, 1000000.0])  # 将每个像素映射到一个唯一值，通过乘k再求和实现
        sum_im = tf.squeeze(tf.reduce_sum(im, axis=1))  # 按行求和并转化为1维
        color_sum, indices = tf.unique(sum_im)  # color_sum：颜色和的种类（1维）,indice:sum_im中数值对应color_sum中的索引
        color_counts = tf.bincount(indices)  # 计算每种索引出现的次数
        color_freq = tf.gather(color_counts, indices)  # 取索引为indices的color_counts中的值，组成一维tensor

        color_freq = tf.reshape(color_freq, [im_shape[0], im_shape[1], 1])  # 形状转化为H * W * 1
        return tf.cast(color_freq, dtype=tf.float32)

    color_freqs = tf.map_fn(get_color_freq, input_high, dtype=tf.float32)  # 对Batch中的每个图执行以上函数
    return color_freqs


def get_color_Fs(images):
    def get_color_F(im):  # 将满足条件在0.0005N - 0.5N的像素位置置1，不满足置0
        N = tf.reduce_prod(tf.shape(im))
        low_bound = 0.00005 * tf.cast(N, tf.float32)
        high_bound = 0.0005 * tf.cast(N, tf.float32)
        F = tf.where(tf.logical_and(im > low_bound, im < high_bound),
                     tf.ones_like(im), tf.zeros_like(im))
        return F

    color_Fs = tf.map_fn(get_color_F, images)
    return color_Fs


def to_attention_points(attention_maps, beta=20):
    """
    Parameters
    ==========
    attention_maps : of teaching data. shape=>BxHxWxC
    binary_point : shape => BxHxW
    """

    def process_attention_map(attention_map):
        binary_point = attention_map[:, :, 0]  # use foreground color
        ones = tf.ones(tf.shape(binary_point))
        zeros = tf.zeros(tf.shape(binary_point))
        binary_point = tf.where(tf.not_equal(binary_point, 0), ones, zeros)
        one_count = tf.reduce_sum(binary_point)

        def true_fn(binary_point):
            indices = tf.where(tf.equal(binary_point, 1))  # 获取值为1的索引
            indices = tf.random_shuffle(indices)[:beta]  # 打乱indices并取前beta个
            updates = tf.ones(tf.shape(indices)[0])
            binary_point = tf.zeros_like(binary_point)
            binary_point = tf.tensor_scatter_nd_update(binary_point, indices, updates)  # 更新binary_point中的值
            return binary_point

        def false_fn(binary_point):
            return binary_point

        # 如果binary_points中1的个数大于beta,随机选择deta个点置1，其余点置0
        binary_point = tf.cond(one_count > beta, lambda: true_fn(binary_point), lambda: false_fn(binary_point))
        attention_points = tf.stack(
            [tf.where(tf.cast(binary_point, tf.bool), attention_map[:, :, 0], tf.zeros_like(attention_map[:, :, 0])),
             tf.where(tf.cast(binary_point, tf.bool), attention_map[:, :, 1],
                      tf.zeros_like(attention_map[:, :, 1]))], axis=-1)
        return tf.cast(binary_point, tf.float32), tf.cast(attention_points, tf.float32)

    binary_points, attention_points = tf.map_fn(process_attention_map, attention_maps, dtype=(tf.float32, tf.float32))
    return binary_points, attention_points


def loss_huber(target, input, delta=0.5, reduction='mean'):
    t = tf.abs(input - target)
    ret = tf.where(t <= delta, 0.5 * (t ** 2), ((delta * t) - (delta ** 2) / 2))
    if reduction != 'none':
        ret = tf.reduce_mean(ret) if reduction == 'mean' else tf.reduce_sum(ret)
    return ret


def loss_mse(ab_out, ab_long):
    mse_loss = tf.reduce_mean(tf.square(ab_long - ab_out))
    return mse_loss


def loss_mae(y_pred, y_true):
    mae_loss = tf.losses.absolute_difference(y_true, y_pred)
    # maes_loss = tf.reduce_sum(maes)
    return mae_loss


class lowlight_enhance(object):
    def __init__(self, sess):
        self.smooth_kernel_y = None
        self.smooth_kernel_x = None
        self.sess = sess
        self.DecomNet_layer_num = 5

        # build the model
        self.input_low = tf.placeholder(tf.float32, [None, None, None, 3], name='input_low')
        self.input_high = tf.placeholder(tf.float32, [None, None, None, 3], name='input_high')

        [R_low, I_low] = DecomNet(self.input_low, layer_num=self.DecomNet_layer_num)
        [R_high, I_high] = DecomNet(self.input_high, layer_num=self.DecomNet_layer_num)

        # CWANab
        R_low_lab = rgb2lab(R_low)
        R_low_l = R_low_lab[:, :, :, 0:1]
        R_low_ab = R_low_lab[:, :, :, 1:3]
        R_low_ab_e, attention_map, attention_points = CWANab(R_low_ab)
        R_high_ab = rgb2lab(R_high)[:, :, :, 1:3]
        F = get_color_Fs(get_color_freqs(self.input_high))  # 此处本应将R_high作为输入参数，但R_high数值不规则会导致map全0，而若强制转换R_high可能造成损失。
        high_attention_map = R_high_ab * F
        high_binary_points, high_attention_points = to_attention_points(high_attention_map)
        high_binary_points = tf.expand_dims(high_binary_points, -1)
        high_binary_points = tf.tile(high_binary_points, [1, 1, 1, 2])
        R_low_new = lab2rgb(concat([R_low_l, R_low_ab_e]))

        I_delta = RelightNet(I_low, R_low)

        I_low_3 = concat([I_low, I_low, I_low])
        I_high_3 = concat([I_high, I_high, I_high])
        I_delta_3 = concat([I_delta, I_delta, I_delta])

        self.output_R_high = R_high
        self.output_high_attention_map = (R_high * F)[:, :, :, 0:1]
        self.output_R_low = R_low
        self.output_I_low = I_low_3
        self.output_I_delta = I_delta_3
        # self.output_S = R_low * I_delta_3
        self.output_S = R_low_new * I_delta_3  # 启用此行可应用CWANab
        # loss
        self.recon_loss_low = tf.reduce_mean(tf.abs(R_low * I_low_3 - self.input_low))
        self.recon_loss_high = tf.reduce_mean(tf.abs(R_high * I_high_3 - self.input_high))
        self.recon_loss_mutal_low = tf.reduce_mean(tf.abs(R_high * I_low_3 - self.input_low))
        self.recon_loss_mutal_high = tf.reduce_mean(tf.abs(R_low * I_high_3 - self.input_high))
        self.equal_R_loss = tf.reduce_mean(tf.abs(R_low - R_high))
        self.relight_loss = tf.reduce_mean(tf.abs(R_low * I_delta_3 - self.input_high))

        self.Ismooth_loss_low = self.smooth(I_low, R_low)
        self.Ismooth_loss_high = self.smooth(I_high, R_high)
        self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        self.loss_huber_out = loss_huber(R_high_ab, R_low_ab_e)
        self.loss_mse_out = loss_mse(attention_points * high_binary_points,
                                     high_attention_points * high_binary_points) / 20.0
        self.loss_ab = self.loss_huber_out + self.loss_mse_out
        self.loss_map = loss_mae(attention_map, high_attention_map)
        self.loss_Decom = self.recon_loss_low + self.recon_loss_high + 0.001 * self.recon_loss_mutal_low + 0.001 * self.recon_loss_mutal_high + 0.1 * self.Ismooth_loss_low + 0.1 * self.Ismooth_loss_high + 0.01 * self.equal_R_loss
        self.loss_Relight = self.relight_loss + 3 * self.Ismooth_loss_delta
        # self.loss_CWANab = self.loss_ab + self.loss_map
        self.loss_CWANab = self.loss_huber_out
        self.lr = tf.placeholder(tf.float32, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        optimizer_CWANab = tf.train.AdamOptimizer(0.0001, 0.05, name='optimizer_CWANab')

        self.var_Decom = [var for var in tf.trainable_variables() if 'DecomNet' in var.name]
        self.var_Relight = [var for var in tf.trainable_variables() if 'RelightNet' in var.name]
        self.var_CWANab = [var for var in tf.trainable_variables() if 'CWANab' in var.name]

        self.train_op_Decom = optimizer.minimize(self.loss_Decom, var_list=self.var_Decom)
        self.train_op_Relight = optimizer.minimize(self.loss_Relight, var_list=self.var_Relight)
        self.train_op_CWANab = optimizer_CWANab.minimize(self.loss_CWANab, var_list=self.var_CWANab)

        self.sess.run(tf.global_variables_initializer())

        self.saver_Decom = tf.train.Saver(var_list=self.var_Decom)
        self.saver_Relight = tf.train.Saver(var_list=self.var_Relight)
        self.saver_CWANab = tf.train.Saver(var_list=self.var_CWANab)

        print("[*] Initialize model successfully...")

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = tf.reshape(tf.constant([[0, 0], [-1, 1]], tf.float32), [2, 2, 1, 1])
        self.smooth_kernel_y = tf.transpose(self.smooth_kernel_x, [1, 0, 2, 3])

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        return tf.abs(tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='SAME'))

    def ave_gradient(self, input_tensor, direction):
        return tf.layers.average_pooling2d(self.gradient(input_tensor, direction), pool_size=3, strides=1,
                                           padding='SAME')

    def smooth(self, input_I, input_R):
        input_R = tf.image.rgb_to_grayscale(input_R)
        return tf.reduce_mean(
            self.gradient(input_I, "x") * tf.exp(-10 * self.ave_gradient(input_R, "x")) + self.gradient(input_I,
                                                                                                        "y") * tf.exp(
                -10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data, sample_dir, train_phase):
        print("[*] Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data)):
            input_low_eval = np.expand_dims(eval_low_data[idx], axis=0)

            if train_phase == "Decom":
                result_1, result_2 = self.sess.run([self.output_R_low, self.output_I_low],
                                                   feed_dict={self.input_low: input_low_eval})
            if train_phase == "Relight":
                result_1, result_2 = self.sess.run([self.output_S, self.output_I_delta],
                                                   feed_dict={self.input_low: input_low_eval})

            save_images(os.path.join(sample_dir, 'eval_%s_%d_%d.png' % (train_phase, idx + 1, epoch_num)), result_1,
                        result_2)

    def train(self, train_low_data, train_high_data, eval_low_data, batch_size, patch_size, epoch, lr, sample_dir,
              ckpt_dir, eval_every_epoch, train_phase):
        assert len(train_low_data) == len(train_high_data)
        numBatch = len(train_low_data) // int(batch_size)

        # load pretrained model
        if train_phase == "Decom":
            train_op = self.train_op_Decom
            train_loss = self.loss_Decom
            saver = self.saver_Decom
        elif train_phase == "Relight":
            train_op = self.train_op_Relight
            train_loss = self.loss_Relight
            saver = self.saver_Relight
        elif train_phase == "CWANab":
            train_op = self.train_op_CWANab
            train_loss = self.loss_CWANab
            saver = self.saver_CWANab

        load_model_status, global_step = self.load(saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training for phase %s, with start epoch %d start iter %d : " % (
            train_phase, start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                # generate data for a batch
                batch_input_low = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_high = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_low_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)

                    rand_mode = random.randint(0, 7)
                    batch_input_low[patch_id, :, :, :] = data_augmentation(
                        train_low_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)
                    batch_input_high[patch_id, :, :, :] = data_augmentation(
                        train_high_data[image_id][x: x + patch_size, y: y + patch_size, :], rand_mode)

                    image_id = (image_id + 1) % len(train_low_data)
                    if image_id == 0:
                        tmp = list(zip(train_low_data, train_high_data))
                        random.shuffle(list(tmp))
                        train_low_data, train_high_data = zip(*tmp)

                # train
                _, loss = self.sess.run([train_op, train_loss],
                                        feed_dict={self.input_low: batch_input_low,
                                                   self.input_high: batch_input_high,
                                                   self.lr: lr[epoch]})
                print("%s Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f"
                      % (train_phase, epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            # evalutate the model and save a checkpoint file for it
            if (epoch + 1) % eval_every_epoch == 0:
                # self.evaluate(epoch + 1, eval_low_data, sample_dir=sample_dir, train_phase=train_phase)
                self.save(saver, iter_num, ckpt_dir, "RetinexNet-%s" % train_phase)

        print("[*] Finish training for phase %s." % train_phase)

    def save(self, saver, iter_num, ckpt_dir, model_name):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        print("[*] Saving model %s" % model_name)
        saver.save(self.sess,
                   os.path.join(ckpt_dir, model_name),
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):
        print(ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(ckpt_dir)
            print(full_path)
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_low_data, test_high_data, test_low_data_names, save_dir, decom_flag):
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status_Decom, _ = self.load(self.saver_Decom, './model/Decom')
        load_model_status_Relight, _ = self.load(self.saver_Relight, './model/Relight')
        load_model_status_CWANab, _ = self.load(self.saver_CWANab, './model/CWANab')

        if load_model_status_Decom and load_model_status_Relight and load_model_status_CWANab:
            print("[*] Load weights successfully...")
        psnr_list = []
        ssim_list = []
        print("[*] Testing...")
        for idx in range(len(test_low_data)):
            print(test_low_data_names[idx])
            [_, name] = os.path.split(test_low_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_low_test = np.expand_dims(test_low_data[idx], axis=0)
            input_high_test = np.expand_dims(test_high_data[idx], axis=0)
            [R_high, high_map_0, R_low, I_low, I_delta, S] = self.sess.run(
                [self.output_R_high, self.output_high_attention_map, self.output_R_low, self.output_I_low, self.output_I_delta,
                 self.output_S],
                feed_dict={self.input_low: input_low_test, self.input_high: input_high_test})
            # print("cat:")
            # print(cat)
            # print("mul:")
            # print(mul)
            myPsnr = psnr(test_high_data[idx], S)
            mySsim = ssim(test_high_data[idx], S)
            psnr_list.append(myPsnr)
            ssim_list.append(mySsim)
            print(
                'idx: ' + str(idx) + '  Image: ' + name + '.' + suffix + '  Psnr: %.5f   Ssim: %.5f' % (myPsnr, mySsim))
            if decom_flag == 1:
                save_images(os.path.join(save_dir, name + "_R_high." + suffix), R_high)
                save_images(os.path.join(save_dir, name + "_R_low." + suffix), R_low)
                save_images(os.path.join(save_dir, name + "_I_low." + suffix), I_low)
                save_images(os.path.join(save_dir, name + "_I_delta." + suffix), I_delta)
                save_images(os.path.join(save_dir, name + "_high_map_0." + suffix), high_map_0)
                # np.set_printoptions(threshold=np.inf)
                # print(R_high)
                # print(high_map_0.shape)
                # im = np.squeeze(high_map_0) * 255.0
                # fig, ax = plt.subplots()
                # plt.imshow(im, cmap='gray')
                # plt.show()

            save_images(os.path.join(save_dir, name + "_S." + suffix), S)
        psnr_mean = np.mean(np.array(psnr_list))
        ssim_mean = np.mean(np.array(ssim_list))
        print('PSNR MEAN: ', psnr_mean)
        print('SSIM MEAN: ', ssim_mean)
        # 获取当前时间
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # 将结果写入文件
        with open("result.txt", "a") as f:
            f.write(f"psnr_mean: {psnr_mean}, ssim_mean: {ssim_mean}, time: {current_time}\n")
