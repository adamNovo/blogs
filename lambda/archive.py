import os
import subprocess
import sys

subprocess.call("""
    zip -r Archive.zip . --exclude=*.DS_Store* --exclude=*__pycache__/* --exclude=*README.md* --exclude=*archive.py* --exclude=*Dockerfile* --exclude=*docker-compose.yml* --exclude=*docker.py*;
    """.format(), shell=True)