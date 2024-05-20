from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow as tf

from model import lowlight_enhance
from utils import *


parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.5, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=20,
                    help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_low_dir', dest='test_low_dir', default='./data/eval15/low',
                    help='directory for testing inputs (low)')
parser.add_argument('--test_high_dir', dest='test_high_dir', default='./data/eval15/high',
                    help='directory for testing inputs (high)')
parser.add_argument('--decom', dest='decom', default=1,
                    help='decom flag, 0 for enhanced results only and 1 for decomposition results')
parser.add_argument('--high_data', dest='high_data', default='./data/all500/high',
                    help='directory for high data')
parser.add_argument('--low_data', dest='low_data', default='./data/all500/low',
                    help='directory for low data')
parser.add_argument('--attentionMap', dest='attentionMap', default='./data/attentionMap',
                    help='directory for attention maps')
args = parser.parse_args()


def lowlight_train(lowlight_enhance, train_low_data_names, train_high_data_names):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    lr = args.start_lr * np.ones([args.epoch])
    lr[20:] = lr[0] / 10.0

    train_low_data = []
    train_high_data = []
    # train_low_data_names = glob('./data/our485/low/*.png')  # + glob('./data/syn/low/*.png')
    # train_high_data_names = glob('./data/our485/high/*.png')  # + glob('./data/syn/high/*.png')
    train_low_data_names.sort()
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('./data/eval/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size,
                           patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
                           ckpt_dir=os.path.join(args.ckpt_dir, 'Decom'), eval_every_epoch=args.eval_every_epoch,
                           train_phase="Decom")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size,
                           patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
                           ckpt_dir=os.path.join(args.ckpt_dir, 'Relight'), eval_every_epoch=args.eval_every_epoch,
                           train_phase="Relight")

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size,
                           patch_size=args.patch_size, epoch=args.epoch, lr=lr, sample_dir=args.sample_dir,
                           ckpt_dir=os.path.join(args.ckpt_dir, 'CWANab'), eval_every_epoch=args.eval_every_epoch,
                           train_phase="CWANab")

    print("模型训练完成")
    src_dirs = ["./checkpoint/CWANab", "./checkpoint/Decom", "./checkpoint/Relight"]
    dest_dirs = ["./model/CWANab", "./model/Decom", "./model/Relight"]
    # src_dirs = ["./checkpoint/Decom", "./checkpoint/Relight"]
    # dest_dirs = ["./model/Decom", "./model/Relight"]
    move_files(src_dirs, dest_dirs)
    print("模型转移完成")


def lowlight_test(lowlight_enhance, test_low_data_name, test_high_data_name):
    if (args.test_low_dir is None) or (args.test_high_dir is None):
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # test_low_data_name = glob(os.path.join(args.test_low_dir) + '/*.*')
    # test_high_data_name = glob(os.path.join(args.test_high_dir) + '/*.*')
    test_low_data = []
    test_high_data = []
    for idx in range(len(test_low_data_name)):
        test_low_im = load_images(test_low_data_name[idx])
        test_low_data.append(test_low_im)
        test_high_im = load_images(test_high_data_name[idx])
        test_high_data.append(test_high_im)
    lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir=args.save_dir,
                          decom_flag=args.decom)


def main(_):
    if os.path.exists('./checkpoint'):
        print('checkpoint目录存在，正在删除以开始实验...')
        shutil.rmtree('./checkpoint')
        print('删除成功，开始实验！')
    high_data = glob(os.path.join(args.high_data) + '/*.png')
    low_data = glob(os.path.join(args.low_data) + '/*.png')
    high_train, high_test, low_train, low_test = split_data(high_data, low_data)
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        # gpu_options.allow_growth = True
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model, low_train, high_train)
                lowlight_test(model, low_test, high_test)
            elif args.phase == 'test':
                low_test = glob(os.path.join(args.test_low_dir) + '/*.*')
                high_test = glob(os.path.join(args.test_high_dir) + '/*.*')
                lowlight_test(model, low_test, high_test)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
    # os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        print("[*] CPU\n")
        with tf.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model, low_train, high_train)
                lowlight_test(model, low_test, high_test)
            elif args.phase == 'test':
                low_test = glob(os.path.join(args.test_low_dir) + '/*.*')
                high_test = glob(os.path.join(args.test_high_dir) + '/*.*')
                lowlight_test(model, low_test, high_test)
            else:
                print('[!] Unknown phase')
                exit(0)


if __name__ == '__main__':
    tf.app.run()

