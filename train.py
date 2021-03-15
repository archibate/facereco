import cv2
import face_recognition
import imutils.paths
import numpy as np
import os.path
import pickle
import sys

def detect_faces(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)
    return boxes, encodings

def match_faces(encodings):
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(data['encodings'], encoding)
        name = 'Unknown'
        if any(matches):
            indices = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in indices:
                name = data['names'][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    return names

def draw_labels(img, boxes, names):
    for (top, right, bottom, left), name in zip(boxes, names):
        cv2.putText(img, name, (left, top - 15 if top > 30 else top + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

if __name__ == '__main__':
    data = []

    imagePaths = list(imutils.paths.list_images(sys.argv[1]))
    for i, imagePath in enumerate(imagePaths):
        print('processing {}/{}: {}'.format(i, len(imagePaths), imagePath))
        name = imagePath.split(os.path.sep)[-2]
        img = cv2.imread(imagePath)
        boxes, encodings = detect_faces(img)
        for encoding in encodings:
            data.append((name, encoding))

    with open('model.pickle', 'wb') as f:
        pickle.dump(data, f)
