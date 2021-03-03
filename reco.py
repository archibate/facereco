import os
import csv
import shutil
import docker
import tempfile
import pickle

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
with open('/tmp/233.pkl', 'wb') as f:
    pickle.dump(res, f)


#__import__('IPython').embed()
