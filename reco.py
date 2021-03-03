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
#print(ret.decode())

with open('/tmp/share/result.csv') as f:
    rows = [[x.strip() for x in row] for row in csv.reader(f)]
    assert len(rows) == 2
    res = dict((k, float(v)) for k, v in zip(*rows))

AUs = 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c'
ret = []
for k in AUs:
    ret.append(res[k])
print(ret)
