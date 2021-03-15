from core import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True,
        help='path to trained model file')
ap.add_argument('-i', '--image', required=True,
        help='path to input image to detect')
ap = vars(ap.parse_args())

dec = FaceDetector.from_trained_model(ap['model'])
img = cv2.imread(ap['image'])
dec.draw_boxes_in_pic(img)
cv2.imshow('face', img)
cv2.waitKey(0)
