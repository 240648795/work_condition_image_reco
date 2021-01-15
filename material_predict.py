# -*- coding : utf-8-*-
import pickle

import pandas as pd
import cv2
import numpy as np
from utils.image_utils.image_feature_utils import get_image_feature


def predict(save_model_path, save_detail_path):
    print("[INFO] 读取模型和标签")
    saved_model = pickle.loads(open(save_model_path, "rb").read())
    saved_detail = pickle.loads(open(save_detail_path, "rb").read())
    return saved_model, saved_detail


def get_prediction_image(img_org, saved_model, saved_detail):
    feature = get_image_feature(img_org, saved_detail.percentile_threshold)
    predict_df = pd.DataFrame([feature])
    preds_proba_list = saved_model.predict_proba(predict_df)
    label = saved_detail.label_encoder.classes_[np.argmax(preds_proba_list)]
    preds_proba = preds_proba_list[0][np.argmax(preds_proba_list)]
    return label, preds_proba


def test_one_image():
    save_model_path = r'save_models/models/material_image_model_old'
    save_detail_path = r'save_models/details/material_image_details_old'
    test_predict_image_path = r'./data/material_image/big_stone/cut_image_col_026f3ff46-4f2b-11eb-a3d9-94e6f7f8d382_row_5.jpg'
    test_predict_image = cv2.imread(test_predict_image_path)
    saved_model, saved_detail = predict(save_model_path=save_model_path, save_detail_path=save_detail_path)
    label, preds_proba = get_prediction_image(test_predict_image, saved_model, saved_detail)
    print(label, preds_proba)


def two_step_prediction(file_path, need_width, need_height, threshold):
    img_org = cv2.imread(file_path)
    img_org = cv2.resize(img_org, (800, 600))
    cols = img_org.shape[1]
    rows = img_org.shape[0]
    need_cols = cols // need_width
    need_rows = rows // need_height

    predicts = []
    for col_i in range(0, need_cols):
        for row_i in range(0, need_rows):
            each_col_start = need_width * col_i
            each_col_end = need_width * col_i + need_width
            each_row_start = need_height * row_i
            each_row_end = need_height * row_i + need_height
            each_pic = img_org[each_row_start:each_row_end, each_col_start:each_col_end, :]
            label, preds_proba = get_prediction_image(each_pic, saved_model, saved_detail)
            predicts.append(label)
            if label != 'unknown':
                if label == 'earth' and preds_proba > threshold:
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 255), 2)
                elif label == 'sand' and preds_proba > threshold:
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 0, 255), 2)
                elif label == 'coal' and preds_proba > threshold:
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (0, 255, 0), 2)
                elif label == 'stone' and preds_proba > threshold:
                    cv2.rectangle(img_org, (each_col_start, each_row_start), (each_col_end, each_row_end),
                                  (255, 255, 255), 3)
                    cv2.putText(img_org, label, (each_col_start + 5, each_row_start + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.4, (255, 0, 0), 2)

    cv2.imshow('work_condition_recognition', img_org)
    cv2.waitKey(0)


save_model_path = r'save_models/models/material_image_model'
save_detail_path = r'save_models/details/material_image_details'
saved_model, saved_detail = predict(save_model_path, save_detail_path)

if __name__ == '__main__':
    # 只测试一张材质图片
    # test_one_image()

    # 整体测试一整正式的图
    # file_path = r'./data/material_image_test/coal2.jpg'
    file_path = r'./data/material_image_test/earth4.png'
    # file_path = r'./data/material_image_test/sand.jpg'
    # file_path = r'./data/material_image_test/stone.jpg'

    two_step_prediction(file_path=file_path, need_width=60, need_height=60, threshold=0.8)
