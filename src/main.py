import cv2 as cv
import numpy as np
import math


def img_rotate(fname, fname_pr):
    image_original = cv.imread(fname, cv.IMREAD_GRAYSCALE)
    
    # выделение границ Кэнни
    image_conf = cv.Canny(image_original, 850, 900)
    cv.imwrite("conf.jpg", image_conf)

    # преобразование Хафа для выделения прямых линий и отсечение линий
    image_conf_lines = cv.cvtColor(image_conf, cv.COLOR_GRAY2BGR)
    lines_angles = []
    lines_hough = cv.HoughLinesP(image_conf, 1, np.pi / 180, 100, None, 20, 10)
    if lines_hough is not None:
        for i in range(0, len(lines_hough)):
            line = lines_hough[i][0]
            if line[2] - line[0] == 0:
                continue
            deg = math.degrees(math.atan((line[3] - line[1]) / (line[2] - line[0])))
            if -40 < deg < 40:
                cv.line(image_conf_lines, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv.LINE_AA)
                lines_angles.append(deg)

    cv.imwrite("image_conf_lines.jpg", image_conf_lines)
    rotation_angle = np.mean(lines_angles)
    print(f"rotation angle = {rotation_angle}")
    (h, w) = image_original.shape[:2]
    center = (w / 2, h / 2)
    image_rotation_matrix = cv.getRotationMatrix2D(center, rotation_angle, 1.0)
    image_rotated = cv.warpAffine(image_original, image_rotation_matrix, (w, h), borderMode=cv.BORDER_REPLICATE)
    cv.imwrite(fname_pr, image_rotated)


if __name__ == "__main__":
    img_rotate("<<Write here input file name>>.jpg", "result.jpg")
