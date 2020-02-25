# TODO: I think 'save_label_from_dcm' only saves one box. Data may have more than one.
# TODO: save the jpeg in 256 * 256 format. Check if runs better than original. Add option not to resize.
# TODO: function to devide to train and validation sets (the two txt files).
# In data/custom/train.txt and data/custom/valid.txt, add paths
# to images that will be used as train and validation data respectively.

import os
from tqdm import tqdm
import cv2
import numpy as np
import pydicom
import pandas as pd


def save_img_from_dcm(dcm_dir, img_dir, patient_id, dsize = (256,256)):
    """
    Converts a 1 channel dcm image to jpg with 3 channels (to allow usage in pretrained).

    :param dcm_dir:
    :param img_dir:
    :param patient_id:
    :param dsize: (int,int)
    """
    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    if os.path.exists(img_fp):
        return
    dcm_fp = os.path.join(dcm_dir, "{}.dcm".format(patient_id))
    img_1ch = pydicom.read_file(dcm_fp).pixel_array
    img_3ch = np.stack([img_1ch] * 3, -1)

    img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
    img_3ch = cv2.resize(img_3ch, dsize, interpolation = cv2.INTER_AREA)
    cv2.imwrite(img_fp, img_3ch)


def save_label_from_dcm(label_dir, patient_id, row=None):
    """
    To prepare for darknet style training, this function creates a txt label file for each patient ID (= each image).
    The row details include
    The output row is formatted as: label_idx x_center y_center width height
    The coordinates should be scaled [0, 1], and the label_idx should be zero-indexed and
    correspond to the row number of the class name in data/custom/classes.names

    :param label_dir:
    :param patient_id:
    :param row:
    """
    # rsna defualt image size
    img_size = 1024 # if the original images have a different size, you must change it here.
    label_fp = os.path.join(label_dir, "{}.txt".format(patient_id))

    f = open(label_fp, "a")
    if row is None:
        f.close()
        return

    top_left_x = row[1]
    top_left_y = row[2]
    w = row[3]
    h = row[4]

    # 'r' means relative. 'c' means center.
    rx = top_left_x / img_size
    ry = top_left_y / img_size
    rw = w / img_size
    rh = h / img_size
    rcx = rx + rw / 2
    rcy = ry + rh / 2

    line = "{} {} {} {} {}\n".format(0, rcx, rcy, rw, rh) # Since RSNA task is binary classification basically, <object-class> is 0.

    f.write(line)
    f.close()


def save_yolov3_data_from_rsna(dcm_dir, img_dir, label_dir, annots, dsize = (256,256), SkipEmpty = True):
    """
    Convert all dcm files into jpg images of some size.
    :param dcm_dir:
    :param img_dir:
    :param label_dir:
    :param annots:
    :param dsize:
    :param SkipEmpty:
    :return:
    """
    for row in tqdm(annots.values):
        patient_id = row[0]

        img_fp = os.path.join(img_dir, "{}.jpg".format(patient_id))
        if os.path.exists(img_fp):
            save_label_from_dcm(label_dir, patient_id, row)
            continue

        target = row[5]
        # Since kaggle kernel have samll volume (5GB ?), I didn't contain files with no bbox here.
        if target == 0 and SkipEmpty:
            continue
        save_label_from_dcm(label_dir, patient_id, row)
        save_img_from_dcm(dcm_dir, img_dir, patient_id, dsize = dsize)

DATA_DIR = r'C:\Users\M\RSNA'

train_dcm_dir = os.path.join(DATA_DIR, "stage_2_train_images")
test_dcm_dir = os.path.join(DATA_DIR, "stage_2_test_images")

img_dir = os.path.join(os.getcwd(), "data/custom/images")  # .jpg
label_dir = os.path.join(os.getcwd(), "data/custom/labels")  # .txt
metadata_dir = os.path.join(os.getcwd(), "metadata") # .txt

for directory in [img_dir, label_dir, metadata_dir]:
    if os.path.isdir(directory):
        continue
    os.mkdir(directory)



annots = pd.read_csv(os.path.join(DATA_DIR, "stage_2_train_labels.csv"))
# annots.head()
print(annots.head())

# label_idx x_center y_center width height
save_yolov3_data_from_rsna(train_dcm_dir, img_dir, label_dir, annots)
save_yolov3_data_from_rsna(train_dcm_dir, img_dir, label_dir, annots)
save_yolov3_data_from_rsna(train_dcm_dir, img_dir, label_dir, annots)
save_yolov3_data_from_rsna(train_dcm_dir, img_dir, label_dir, annots)
