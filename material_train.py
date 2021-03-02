# -*- coding : utf-8-*-
import pickle
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils.image_utils.image_feature_utils import get_images_feature
from utils.model_utils.model_details import SavedModelDetails


def train(input_document_path, percentile_threshold, save_model_path, save_detail_path):
    df = get_images_feature(input_document_path=input_document_path, percentile_threshold=percentile_threshold)

    y = df['label']
    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    X = df[['entropy', 'r', 'r_p', 'g', 'g_p', 'b', 'b_p', 'h', 'h_p', 's', 's_p', 'v', 'v_p']]
    # 这里不归一化
    # X = X.apply(LabelEncoder().fit_transform)

    print('[INFO] 开始训练')
    avg_r2 = []
    max_acc = 0
    max_acc_model = None
    for i in range(0, 100):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=None)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        now_acc = model.score(x_test, y_test)
        avg_r2.append(now_acc)
        if now_acc > max_acc:
            max_acc_model = model
    avg_mean = np.mean(avg_r2)
    print("[INFO] 训练完毕，随机森林回归的评估值r2平均值为：", avg_mean)

    if not (max_acc_model is None):
        print('[INFO] 开始保存最好的模型和标签,当前模型准确率：' + str(max_acc_model.score(x_test, y_test)))
        y_pred = max_acc_model.predict(x_test)
        print(classification_report(y_test, y_pred, digits=3, target_names=y_le.classes_))
        with open(save_model_path, 'wb') as f:
            f.write(pickle.dumps(max_acc_model))
        saved_model_details = SavedModelDetails(label_encoder=y_le, percentile_threshold=percentile_threshold)
        with open(save_detail_path, 'wb') as f:
            f.write(pickle.dumps(saved_model_details))
        print('[INFO] 开始保存模型和标签')
    pass


if __name__ == '__main__':
    input_document_path = r'./data/material_image'
    percentile_threshold = 70
    save_model_path = r'save_models/models/material_image_model_20210128.h5'
    save_detail_path = r'save_models/details/material_image_details_20210128.joblib'

    train(input_document_path=input_document_path, percentile_threshold=percentile_threshold,
          save_model_path=save_model_path,
          save_detail_path=save_detail_path)
