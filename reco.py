#__import__('IPython').embed()
import os, docker
C = docker.from_env()

try:
    os.mkdir('/tmp/share')
except IOError:
    pass

def wrapcmd(x):
    return f'-c "{x}"'

ret = C.containers.run('algebr/openface',
        wrapcmd('echo a > /mnt/share/a.txt'),
        detach=True,
        volumes={
            '/tmp/share': dict(bind='/mnt/share', mode='rw'),
        })
print(ret.logs().decode())
