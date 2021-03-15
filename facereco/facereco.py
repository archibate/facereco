import cv2
import face_recognition
import imutils.paths
import numpy as np
import os.path
import pickle
import time

'''
pip3 install opencv-python face_recognition imutils joblib numpy dlib
'''

def encode(img, shift=0):
    rgb_down = rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(shift):
        rgb_down = cv2.pyrDown(rgb_down)
    boxes = list(face_recognition.face_locations(rgb_down))
    for i, (top, right, bottom, left) in enumerate(boxes):
        top <<= shift
        right <<= shift
        bottom <<= shift
        left <<= shift
        boxes[i] = top, right, bottom, left
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings

def detect(data, img, shift=0):
    encodings = encode(img)
    for encoding in encodings:
        matches = face_recognition.compare_faces(data, encoding)
        if any(matches):
            return True
    return False

def train(images, parallel=False):
    data = []
    images = list(images)
    t0 = time.time()
    if parallel:
        import joblib
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), len(images))
        parallel = joblib.Parallel(n_jobs=n_jobs)
        iterator = parallel(joblib.delayed(encode)(img) for img in images)
        for encodings in iterator:
            data.extend(encodings)
    else:
        for i, img in enumerate(images):
            print('training {}/{}'.format(i + 1, len(images)))
            encodings = encode(img)
            data.extend(encodings)
    print(time.time() - t0, 'secs')
    return data


if __name__ == '__main__':
    data = train(map(cv2.imread, imutils.paths.list_images('dataset/linus/')))
    print(detect(data, cv2.imread('images/linus.jpg')))
