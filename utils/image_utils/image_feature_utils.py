# -*- coding : utf-8-*-
import math
import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.image_utils import image_load_utils


def get_image_entropy(img_org):
    tmp = []
    for i in range(256):
        tmp.append(0)
    val = 0
    k = 0
    res = 0
    image = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(image, 9)
    image = cv2.medianBlur(image, 9)
    img = np.array(image)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k = float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if (tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res


def get_image_rgb_hsv(img_org, percentile_threshold):
    # filepath = filepath
    # img_org = cv2.imread(filepath)
    img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    r_hist = cv2.calcHist([img_org_rgb], [0], None, [256], [0, 255])
    max_r = np.argmax(r_hist).astype('int')
    percentile_r = np.percentile(r_hist, percentile_threshold).astype('int')

    g_hist = cv2.calcHist([img_org_rgb], [1], None, [256], [0, 255])
    max_g = np.argmax(g_hist).astype('int')
    percentile_g = np.percentile(g_hist, percentile_threshold).astype('int')

    b_hist = cv2.calcHist([img_org_rgb], [2], None, [256], [0, 255])
    max_b = np.argmax(b_hist).astype('int')
    percentile_b = np.percentile(b_hist, percentile_threshold).astype('int')

    img_org_hsv = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)

    h_hist = cv2.calcHist([img_org_hsv], [0], None, [360], [0, 360])
    max_h = np.argmax(h_hist).astype('int')
    percentile_h = np.percentile(h_hist, percentile_threshold).astype('int')

    s_hist = cv2.calcHist([img_org_hsv], [1], None, [100], [0, 100])
    max_s = np.argmax(s_hist).astype('int')
    percentile_s = np.percentile(s_hist, percentile_threshold).astype('int')

    v_hist = cv2.calcHist([img_org_hsv], [2], None, [100], [0, 100])
    max_v = np.argmax(v_hist).astype('int')
    percentile_v = np.percentile(v_hist, percentile_threshold).astype('int')

    return max_r, percentile_r, \
           max_g, percentile_g, \
           max_b, percentile_b, \
           max_h, percentile_h, \
           max_s, percentile_s, \
           max_v, percentile_v


def get_image_feature(img_org, percentile_threshold):
    entropy = get_image_entropy(img_org=img_org)
    max_r, percentile_r, max_g, percentile_g, \
    max_b, percentile_b, max_h, percentile_h, \
    max_s, percentile_s, max_v, percentile_v = get_image_rgb_hsv(
        img_org=img_org, percentile_threshold=percentile_threshold)

    feature = [entropy,
               max_r, percentile_r,
               max_g, percentile_g,
               max_b, percentile_b,
               max_h, percentile_h,
               max_s, percentile_s,
               max_v, percentile_v]
    return feature


def get_images_feature(input_document_path, percentile_threshold):
    input_document_path = input_document_path
    imagePaths = sorted(list(image_load_utils.list_images(input_document_path)))
    features = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        img_org = cv2.imread(imagePath)
        each_feature = get_image_feature(img_org=img_org, percentile_threshold=percentile_threshold)
        each_feature.append(label)
        features.append(each_feature)
    df = pd.DataFrame(features)
    df.columns = ['entropy', 'r', 'r_p', 'g', 'g_p', 'b', 'b_p', 'h', 'h_p', 's', 's_p', 'v', 'v_p', 'label']
    return df


if __name__ == '__main__':
    # 测试一个图片的特征
    # input = r'./conv_test_image/red_earth.jpg'
    # output = r'./image_feature_csv/image_feature.cvs'
    # each_feature = get_image_feature(input=input, percentile_threshold=90)
    # each_feature.append('1')

    input_document_path = r'../../data/material_image'
    df = get_images_feature(input_document_path=input_document_path, percentile_threshold=90)
    print(df)

    pass
