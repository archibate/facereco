from core import FaceDetector
import argparse
import cv2

def benchmark(func, times=20, warmup=2):
    import time
    for i in range(warmup):
        t = time.time()
        func()
        print('warmup', time.time() - t)
    t0 = time.time()
    for i in range(times):
        t = time.time()
        func()
        print('running', time.time() - t)
    t1 = time.time()
    dt = (t1 - t0) / times
    print('average', dt)

dec = FaceDetector.from_trained_model('model.pickle')
img = cv2.imread('testdata/nvidia.jpg')
benchmark(lambda: dec.is_name_in_pic('nvidia', img))
