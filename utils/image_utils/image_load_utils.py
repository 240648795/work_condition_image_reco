import os
import sys
import cv2 as cv
import os
import shutil
import cv2
import uuid

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def cut_pic_into_feature(file_path, save_image_path, need_width, need_height, label):
    img_org = cv2.imread(file_path)
    cols = img_org.shape[1]
    rows = img_org.shape[0]

    need_cols = cols // need_width
    need_rows = rows // need_height

    for col_i in range(0, need_cols):
        for row_i in range(0, need_rows):
            each_col_start = need_width * col_i
            each_col_end = need_width * col_i + need_width
            each_row_start = need_height * row_i
            each_row_end = need_height * row_i + need_height
            each_pic = img_org[each_row_start:each_row_end, each_col_start:each_col_end, :]
            if each_pic is None:
                print('col_i:' + str(col_i) + ' row_i:' + str(row_i) + ' is None')
            else:
                print('col_i:' + str(col_i) + ' row_i:' + str(row_i) + ' is None')
                cv2.imwrite(save_image_path + '/' + label + '_' +
                            str(uuid.uuid1()) + '.jpg',
                            each_pic)


if __name__ == '__main__':
    input_file_path = r'../../data/need_cut_image'
    save_file_path = r'../../data/save_cut_image'
    imagePaths = sorted(list(list_images(input_file_path)))
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        save_image_path = save_file_path + '/' + label
        need_width = 60
        need_height = 60
        cut_pic_into_feature(imagePath, save_image_path, need_width, need_height, label)

    pass
