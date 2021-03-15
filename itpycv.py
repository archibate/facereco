import cv2
import face_recognition
import numpy as np

img = cv2.imread('dataset/europian/001.jpg')
#img = cv2.imread('dataset/doubles/233.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

for top, right, bottom, left in boxes:
    name = 'europian'
    cv2.putText(img, name, (left, top - 15 if top > 30 else top + 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow('face', img)
cv2.waitKey(0)
