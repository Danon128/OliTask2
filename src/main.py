import cv2 as cv
import numpy as np
import math

def img_rotate(fname, fname_pr):
    image_original = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    final = cv.imread(fname, cv.IMREAD_COLOR)

    #очистка от шумов
    image_clean = cv.adaptiveThreshold(image_original, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 10)
    cv.imwrite('clean.jpg', image_clean)
    
    #изменение размера
    final_wide = 200
    r = float(final_wide) / image_original.shape[1]
    dim = (final_wide, int(image_original.shape[0] * r))
    image_resized = cv.resize(image_clean, dim, interpolation = cv.INTER_AREA)
    cv.imwrite('resized.jpg', image_resized)

    #выделение границ Кэнни
    image_conf = cv.Canny(image_original, 850, 900)
    cv.imwrite('conf.jpg', image_conf)

    cdstP = cv.cvtColor(image_conf, cv.COLOR_GRAY2BGR)

    #преобразование Хафа и отсечение линий
    lines_angles = []
    linesP = cv.HoughLinesP(image_conf, 1, np.pi / 180, 100, None, 20, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if l[2]-l[0] == 0:
                continue;
            deg = math.degrees(math.atan((l[3]-l[1])/(l[2]-l[0])))
            if -40 < deg < 40:
                cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv.LINE_AA) 
                lines_angles.append(deg)

    cv.imwrite('dstWithLinesP.jpg', cdstP)
    rotation_angle = np.mean(lines_angles)
    print(f'rotation angle = {rotation_angle}')
    (h, w) = image_original.shape[:2]
    center = (w / 2, h / 2)
    M = cv.getRotationMatrix2D(center, rotation_angle, 1.0)
    binrotated = cv.warpAffine(image_original, M, (w, h), borderMode=cv.BORDER_REPLICATE)
    rotated = cv.warpAffine(final, M, (w, h), borderMode=cv.BORDER_REPLICATE)

    cv.imwrite(fname_pr, rotated)
