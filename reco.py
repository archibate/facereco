import os, docker
cli = docker.APIClient()
print(cli.version())
