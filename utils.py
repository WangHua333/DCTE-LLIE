import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil


def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


def load_images(file):
    im = Image.open(file)
    return np.array(im, dtype="float32") / 255.0
# def load_images(file, type):
#     if type == "high":
#         path = "./data/our485/high"
#     elif type == "low":
#         path = "./data/our485/low"
#     else:
#         raise ValueError("Invalid type, should be 'high' or 'low'")
#     file_path = os.path.join(path, file)
#     im = Image.open(file_path)
#     return np.array(im, dtype="float32") / 255.0


def save_images(filepath, result_1, result_2=None):
    result_1 = np.squeeze(result_1)
    result_2 = np.squeeze(result_2)

    if not result_2.any():
        cat_image = result_1
    else:
        cat_image = np.concatenate([result_1, result_2], axis=1)

    im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
    im.save(filepath, 'png')


def split_data(high_data, low_data, test_size=0.2):
    # high_images = os.listdir(high_path)
    # low_images = os.listdir(low_path)
    high_train, high_test, low_train, low_test = train_test_split(high_data, low_data, test_size=test_size,
                                                                  random_state=42)
    return high_train, high_test, low_train, low_test


def move_files(src_dirs, dest_dirs):
    # Find the largest num in the filenames
    for src_dir, dest_dir in zip(src_dirs, dest_dirs):
        num = 0
        for filename in os.listdir(src_dir):
            if "RetinexNet-" in filename:
                file_num = int((filename.split("-")[2]).split(".")[0])
                if file_num > num:
                    num = file_num

        # Copy the three files with the largest num
        for suffix in ["data-00000-of-00001", "meta", "index"]:
            src_file = f"RetinexNet-{os.path.basename(src_dir)}-{num}.{suffix}"
            src_path = os.path.join(src_dir, src_file)
            dest_file = f"RetinexNet-tensorflow.{suffix}"
            dest_path = os.path.join(dest_dir, dest_file)
            shutil.copy2(src_path, dest_path)
            print("已将 " + src_file + " 移动至 " + dest_path)
