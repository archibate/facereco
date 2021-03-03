#__import__('IPython').embed()
import os
import csv
import shutil
import docker
import tempfile
import numpy as np

def process_image(imgpath):
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copyfile(imgpath, tmpdir + '/image.jpg')

        def wrapcmd(x):
            return f'-c "{x}"'

        C = docker.from_env()
        stdout = C.containers.run('algebr/openface',
                wrapcmd('build/bin/FaceLandmarkVidMulti -f /mnt/share/image.jpg'
                        ' && cp processed/image.csv /mnt/share/result.csv'),
                volumes={
                    tmpdir: dict(bind='/mnt/share', mode='rw'),
                }, auto_remove=True)

        with open(tmpdir + '/result.csv') as f:
            rows = [[x.strip() for x in row] for row in csv.reader(f)]
            assert len(rows) == 2
            res = dict((k, float(v)) for k, v in zip(*rows))

    return res, rows[0]


def get_image_ident(imgpath):
    print('processing', imgpath)
    res, keys = process_image(imgpath)
    keys = 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'
    ret = []
    for k in keys:
        ret.append(res[k])
    return np.array(ret)


def mse(x, y):
    return np.linalg.norm(x - y)


a = get_image_ident('001.jpg')
b = get_image_ident('030.jpg')
c = get_image_ident('233.jpg')
d = get_image_ident('666.jpg')

print('001=030', mse(a, b))
print('233=666', mse(c, d))
print('001-233', mse(a, c))
print('030-233', mse(b, c))
print('001-666', mse(a, d))
print('030-666', mse(b, d))
