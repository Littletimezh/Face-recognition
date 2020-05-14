import cv2 as cv
import os


class_path = "data\\face5\\"
for img_name in os.listdir(class_path):
    img_path = class_path + img_name
    dst = cv.imread(img_path)
    img = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    cv.imwrite(img_name, img)