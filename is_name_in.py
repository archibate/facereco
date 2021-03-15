from core import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
        help='path to trained model file')
ap.add_argument('-i', '--image', required=True,
        help='path to input image to detect')
ap.add_argument('-n', '--name', required=True,
        help='name of the face to query')
ap = vars(ap.parse_args())

dec = FaceDetector.from_trained_model(ap['model'])
img = cv2.imread(ap['image'])
result = dec.is_name_in_pic(ap['name'], img)
print('YES' if result else 'NO')
