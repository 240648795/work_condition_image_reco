# -*- coding : utf-8-*-
import cv2
import numpy as np


def get_earth_edge(img_org):
    img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_org_rgb = cv2.medianBlur(img_org_rgb, 9)
    img_org_rgb = cv2.medianBlur(img_org_rgb, 9)

    lower_blue = np.array([25, 0, 0])
    upper_blue = np.array([32, 10, 10])
    mask = cv2.inRange(img_org_rgb, lower_blue, upper_blue)

    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    canny = cv2.Canny(dilate, 50, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    show_contours = []
    contours_area = []

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00'] > 100:
            show_contours.append(contour)
            contours_area.append(M['m00'])

    if len(show_contours) > 0:
        contours_area_id = contours_area.index(max(contours_area))
        max_area_contours = show_contours[contours_area_id]
        M = cv2.moments(max_area_contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.drawContours(img_org, show_contours, -1, (255, 0, 0), 2)
        cv2.putText(img_org, "earth", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_org


def get_stone_edge(img_org):
    img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_org_rgb = cv2.medianBlur(img_org_rgb, 9)
    img_org_rgb = cv2.medianBlur(img_org_rgb, 9)

    lower_blue = np.array([60, 68, 25])
    upper_blue = np.array([180, 90, 67])
    mask = cv2.inRange(img_org_rgb, lower_blue, upper_blue)

    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    canny = cv2.Canny(dilate, 50, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_x_list = []
    min_x_list = []
    max_y_list = []
    min_y_list = []

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00'] > 0:
            max_rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(max_rect)
            box = np.int0(box)
            max_x = np.max(box[:, 0])
            min_x = np.min(box[:, 0])
            max_y = np.max(box[:, 1])
            min_y = np.min(box[:, 1])
            max_x_list.append(max_x)
            min_x_list.append(min_x)
            max_y_list.append(max_y)
            min_y_list.append(min_y)

    if len(max_x_list) > 0 and len(min_x_list) > 0 and len(max_y_list) > 0 and len(min_y_list) > 0:
        cv2.rectangle(img_org, (np.min(min_y_list), np.min(min_y_list)), (np.max(max_x_list), np.max(max_y_list)),
                      (255, 0, 0), 0)

        cv2.putText(img_org, "stone",
                    (int((np.min(min_x_list) + np.max(max_x_list)) / 2),
                     int((np.min(min_y_list) + np.max(max_y_list)) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_org


def get_sand_edge(img_org):
    img_org_rgb = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_org_rgb = cv2.medianBlur(img_org_rgb, 11)

    # lower_blue = np.array([28, 14, 17])
    # upper_blue = np.array([41, 30, 20])
    lower_blue = np.array([30, 12, 17])
    upper_blue = np.array([45, 38, 25])
    mask = cv2.inRange(img_org_rgb, lower_blue, upper_blue)
    mask = cv2.medianBlur(mask, 11)

    erode = cv2.erode(mask, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)

    canny = cv2.Canny(dilate, 50, 150)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    show_contours = []
    contours_area = []

    for i, contour in enumerate(contours):
        M = cv2.moments(contour)
        if M['m00'] > 200:
            show_contours.append(contour)
            contours_area.append(M['m00'])

    if len(show_contours) > 0:
        contours_area_id = contours_area.index(max(contours_area))
        max_area_contours = show_contours[contours_area_id]
        M = cv2.moments(max_area_contours)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.drawContours(img_org, [max_area_contours], -1, (255, 0, 0), 2)
        cv2.putText(img_org, "sand", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_org


def get_earth_edge_video():
    vc = cv2.VideoCapture(r'./data/test_video/huangtu.mp4')  # 读入视频文件
    c = 0  # 计数  统计对应帧号
    rval = vc.isOpened()  # 判断视频是否打开  返回True或Flase
    width = 64 * 10
    height = 32 * 10

    # 视频保存配置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_out = cv2.VideoWriter('./data/generate_video/earth_recognition.mp4', fourcc, 20,
                              (width, height))
    while rval:
        rval, frame = vc.read()
        if rval:
            image = frame.copy()
            image_height = image.shape[0]
            image_width = image.shape[1]
            image_left = image[0:image_height, 0:image_width // 2, :]
            image_left_resized = cv2.resize(image_left, (width, height), interpolation=cv2.INTER_AREA)
            image_right = image[0:image_height, image_width // 2:image_width, :]
            image_right_resized = cv2.resize(image_right, (width, height), interpolation=cv2.INTER_AREA)

            image_left_resized = get_earth_edge(image_left_resized)
            # 视频可视化
            # cv2.imshow('work_condition_recognition', image_left_resized)
            # 视频保存
            img_out.write(image_left_resized)
            key = cv2.waitKey(10)
        else:
            break
    vc.release()


def get_stone_edge_video():
    vc = cv2.VideoCapture(r'./data/test_video/shikuai.mp4')  # 读入视频文件
    c = 0  # 计数  统计对应帧号
    rval = vc.isOpened()  # 判断视频是否打开  返回True或Flase
    width = 64 * 10
    height = 32 * 10

    # 视频保存配置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_out = cv2.VideoWriter('./data/generate_video/stone_recognition.mp4', fourcc, 20,
                              (width, height))
    while rval:
        rval, frame = vc.read()
        if rval:
            image = frame.copy()
            image_height = image.shape[0]
            image_width = image.shape[1]
            image_left = image[0:image_height, 0:image_width // 2, :]
            image_left_resized = cv2.resize(image_left, (width, height), interpolation=cv2.INTER_AREA)
            image_right = image[0:image_height, image_width // 2:image_width, :]
            image_right_resized = cv2.resize(image_right, (width, height), interpolation=cv2.INTER_AREA)

            image_left_resized = get_stone_edge(image_left_resized)
            # 视频可视化
            # cv2.imshow('work_condition_recognition', image_left_resized)
            # 视频保存
            img_out.write(image_left_resized)
            key = cv2.waitKey(10)
        else:
            break
    vc.release()


def get_sand_edge_video():
    vc = cv2.VideoCapture(r'./data/test_video/sisha.mp4')  # 读入视频文件
    c = 0  # 计数  统计对应帧号
    rval = vc.isOpened()  # 判断视频是否打开  返回True或Flase
    width = 64 * 10
    height = 32 * 10

    # 视频保存配置
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_out = cv2.VideoWriter('./data/generate_video/sand_recognition.avi', fourcc, 20,
                              (width, height))
    while rval:
        rval, frame = vc.read()
        if rval:
            image = frame.copy()
            image_height = image.shape[0]
            image_width = image.shape[1]
            image_left = image[0:image_height, 0:image_width // 2, :]
            image_left_resized = cv2.resize(image_left, (width, height), interpolation=cv2.INTER_AREA)
            image_right = image[0:image_height, image_width // 2:image_width, :]
            image_right_resized = cv2.resize(image_right, (width, height), interpolation=cv2.INTER_AREA)

            image_left_resized = get_sand_edge(image_left_resized)
            # 视频可视化
            # cv2.imshow('work_condition_recognition', image_left_resized)
            # key = cv2.waitKey(10)
            # 视频保存
            img_out.write(image_left_resized)
        else:
            break
    vc.release()


if __name__ == '__main__':
    # get_earth_edge_video()
    # get_stone_edge_video()
    get_sand_edge_video()
    pass
