import cv2

face_patterns = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_patterns.load('C:\Users\Administrator\Desktop\shujuku\bao\haarcascade_frontalface_default.xml')
image = cv2.imread('group.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_patterns.detectMultiScale(image, scaleFactor=1.00025, minNeighbors=5, minSize=(100, 100))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imwrite('result.jpg', image)
cv2.imshow('output', image)
cv2.waitKey(0)