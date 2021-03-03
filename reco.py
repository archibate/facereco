import os, docker
C = docker.from_env()

ret = C.containers.run('algebr/openface:latest', '-c "ls -l"')
print(ret.decode())

#__import__('IPython').embed()
