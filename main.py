import cv2
import face_recognition
import imutils.paths
import numpy as np
import os.path
import pickle
import time

class FaceDetector:
    def __init__(self, data=(), shift=0):
        self.data = list(data)
        self.shift = shift

    def encode(self, img):
        rgb_down = rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(self.shift):
            rgb_down = cv2.pyrDown(rgb_down)
        boxes = list(face_recognition.face_locations(rgb_down))
        for i, (top, right, bottom, left) in enumerate(boxes):
            top <<= self.shift
            right <<= self.shift
            bottom <<= self.shift
            left <<= self.shift
            boxes[i] = top, right, bottom, left
        encodings = face_recognition.face_encodings(rgb, boxes)
        return encodings

    def detect(self, img):
        encodings = self.encode(img)
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.data, encoding)
            if any(matches):
                return True
        return False

    def train(self, paths):
        self.data = []
        paths = list(paths)
        for i, path in enumerate(paths):
            print('training {}/{}: {}'.format(i + 1, len(paths), path))
            img = cv2.imread(path)
            encodings = self.encode(img)
            self.data.extend(encodings)
        return self.data


if __name__ == '__main__':
    dec = FaceDetector()
    dec.train(imutils.paths.list_images('dataset/linus/'))
    print(dec.detect(cv2.imread('images/linus.jpg')))
