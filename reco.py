#__import__('IPython').embed()
import os, docker
C = docker.from_env()

def wrapcmd(x):
    return f'-c "{x}"'

ret = C.containers.run('algebr/openface',
        wrapcmd('echo hello world'),
        detach=True)
print(ret.logs().decode())
