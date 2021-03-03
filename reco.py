#__import__('IPython').embed()
import os, shutil, docker, csv
C = docker.from_env()

shutil.rmtree('/tmp/share', ignore_errors=True)
os.mkdir('/tmp/share')

shutil.copyfile('./001.jpg', '/tmp/share/image.jpg')

def wrapcmd(x):
    return f'-c "{x}"'

ret = C.containers.run('algebr/openface',
        wrapcmd('build/bin/FaceLandmarkVidMulti -f /mnt/share/image.jpg'
                ' && cp processed/image.csv /mnt/share/result.csv'),
        volumes={
            '/tmp/share': dict(bind='/mnt/share', mode='rw'),
        }, auto_remove=True)
print(ret.decode())

with open('/tmp/share/result.csv') as f:
    rows = [[x.strip() for x in row] for row in csv.reader(f)]
    assert len(rows) == 2
    res = dict((k, float(v)) for k, v in zip(*rows))
    print(res)
