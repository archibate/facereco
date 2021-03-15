import cv2
import face_recognition
import numpy as np
import pickle
import time

with open('model.pickle', 'rb') as f:
    data = pickle.load(f)

def detect_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    return boxes, encodings

def match_faces(encodings, filters=None):
    names = []
    data_names = []
    data_encodings = []
    for name, encoding in data:
        if filters is None or name in filters:
            data_names.append(name)
            data_encodings.append(encoding)
    for encoding in encodings:
        matches = face_recognition.compare_faces(data_encodings, encoding)
        name = 'unknown'
        if any(matches):
            indices = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in indices:
                name = data_names[i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    return names

def draw_labels(img, boxes, names):
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.putText(img, name, (left, top - 15 if top > 30 else top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

t0 = time.time()
img = cv2.imread('nvidia.jpg')
boxes, encodings = detect_faces(img)
names = match_faces(encodings)
draw_labels(img, boxes, names)
print(time.time() - t0)
cv2.imshow('face', img)
cv2.waitKey(0)
