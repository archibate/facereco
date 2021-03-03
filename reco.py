import os
import csv
import shutil
import docker
import tempfile
import numpy as np
import matplotlib.pyplot as plt

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
            keys = rows[0]

    return res, keys


res, keys = process_image('233.jpg')



pos = []
for i in range(100):
     if f'x_{i}' not in res:
         break
     pos.append((res[f'x_{i}'], res[f'y_{i}']))
pos = np.array(pos)

plt.scatter(pos[:, 0], pos[:, 1])
plt.show()


#__import__('IPython').embed()
