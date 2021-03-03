#__import__('IPython').embed()
import os, shutil, docker
C = docker.from_env()

shutil.rmtree('/tmp/share', ignore_errors=True)
os.mkdir('/tmp/share')

shutil.copyfile('./001.jpg', '/tmp/share/image.jpg')
shutil.copyfile('./main.sh', '/tmp/share/main.sh')

def wrapcmd(x):
    return f'-c "{x}"'

ret = C.containers.run('algebr/openface',
        wrapcmd('bash /mnt/share/main.sh'),
        volumes={
            '/tmp/share': dict(bind='/mnt/share', mode='rw'),
        }, auto_remove=True)
print(ret.decode())

with open('/tmp/share/result.csv') as f:
    print(f.read())
