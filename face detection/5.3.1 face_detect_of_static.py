# 导入相关库
import cv2

# 要检测的人脸图片
filename = "face_data/6.jpg"


# 定义人脸检测函数
def detect(filename):
    # 定义haar级联特征分类器对象
    face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

    img = cv2.imread(filename=filename)

    # RGB图像灰度化。因为haar级联人脸检测需要灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用haar级联特征分类器检测人脸
    # 参数scaleFactor(必选)：表示人脸检测过程中每次迭代时图像的压缩率；
    # 参数minNeighbors(必选)：表示每个人脸矩形保留近邻数目的最小值；
    # 返回：人脸矩形数组(x, y, w, h),(x,y)表示左上角的坐标，(w和h)表示人脸矩形的宽度和高度。
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

    # 根据返回的矩形数组(x,y,w,h)在原图上画矩形框
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=2)

    cv2.namedWindow("Detected!!")
    cv2.imshow("Detected!!", img)
    cv2.imwrite("face_data\\6.1.jpg", img=img)
    cv2.waitKey(0)


detect(filename)
