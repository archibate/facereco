import cv2
import face_recognition
import imutils.paths
import numpy as np
import os.path
import pickle
import time

class FaceDetector:
    def __init__(self, data, shift=0):
        self.data = list(data)
        self.shift = shift

    @classmethod
    def train_from_dataset(cls, path, shift=0):
        self = cls([], shift)
        self._train_dataset(path)
        return self

    @classmethod
    def from_trained_model(cls, path, shift=0):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self = cls(data, shift)
        return self

    def save_trained_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)

    def _detect_faces(self, img):
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
        return boxes, encodings

    def _match_faces(self, encodings, filters=None):
        names = []
        data_names = []
        data_encodings = []
        for name, encoding in self.data:
            if not filters or name in filters:
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

    def _draw_labels(self, img, boxes, names):
        for (top, right, bottom, left), name in zip(boxes, names):
            cv2.putText(img, name, (left, top - 15 if top > 30 else top + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    def _train_dataset(self, path):
        self.data = []
        imagePaths = list(imutils.paths.list_images(path))
        for i, imagePath in enumerate(imagePaths):
            print('training {}/{}: {}'.format(i + 1, len(imagePaths), imagePath))
            name = imagePath.split(os.path.sep)[-2]
            img = cv2.imread(imagePath)
            boxes, encodings = self._detect_faces(img)
            for encoding in encodings:
                self.data.append((name, encoding))
        return self.data

    def is_name_in_pic(self, name, img):
        boxes, encodings = self._detect_faces(img)
        names = self._match_faces(encodings, [name])
        exists = name in names
        return exists

    def get_names_in_pic(self, img):
        boxes, encodings = self._detect_faces(img)
        names = self._match_faces(encodings)
        return names

    def draw_boxes_in_pic(self, img):
        boxes, encodings = self._detect_faces(img)
        names = self._match_faces(encodings)
        self._draw_labels(img, boxes, names)
