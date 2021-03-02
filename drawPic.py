# -*- coding : utf-8-*-
import numpy as np
import cv2 as cv

if __name__ == '__main__':
    # pinghualeng = [i + 1 for i in range(0, 200)]
    # pinghua = []
    #
    # for i in range(0, 100):
    #     pinghua.append(pinghualeng)
    #
    # pinghua_np = np.asarray(pinghua,np.uint8)
    #
    # print(pinghua_np)
    #
    # cv.imwrite('pinghua.jpg',pinghua_np)

    # bupinghualeng = [i + 1 for i in range(0, 200)]
    #
    # bupinghualeng[99] = 200
    # bupinghualeng[100] = 200
    # bupinghualeng[101] = 200
    # bupinghualeng[102] = 200
    #
    # bupinghua = []
    # for i in range(0, 100):
    #     bupinghua.append(bupinghualeng)
    #
    # bupinghua_np = np.asarray(bupinghua, np.uint8)
    #
    # cv.imwrite('bupinghualeng.jpg', bupinghua_np)


    bupinghualeng = [i + 1 for i in range(0, 200)]

    bupinghualeng[99] = 200
    bupinghualeng[100] = 200
    bupinghualeng[101] = 200
    bupinghualeng[102] = 200

    bupinghua = []
    for i in range(0, 100):
        bupinghua.append(bupinghualeng)

    bupinghua_np = np.asarray(bupinghua, np.uint8)

    x = cv.Sobel(bupinghua_np, cv.CV_16S, 1, 0)
    y = cv.Sobel(bupinghua_np, cv.CV_16S, 0, 1)

    absX = cv.convertScaleAbs(x)  # 转回uint8
    absY = cv.convertScaleAbs(y)

    dst = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv.imwrite('sobelabsX.jpg', absX)
    cv.imwrite('sobelabsY.jpg', absY)
    cv.imwrite('sobers.jpg', dst)


