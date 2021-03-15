from core import FaceDetector
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
        help='path to input directory of images')
ap.add_argument('-m', '--model', required=True,
        help='path to output model file')
ap = vars(ap.parse_args())

dec = FaceDetector.train_from_dataset(ap['dataset'])
dec.save_trained_model(ap['model'])
