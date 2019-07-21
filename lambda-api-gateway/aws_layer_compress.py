import os
import subprocess
import sys

cmd = """
zip -r python.zip ./python 
--exclude=*.DS_Store* 
--exclude=*__pycache__/* 
--exclude=*.pyc*;
"""

cmd = "".join(cmd.splitlines()) # remove newlines
subprocess.call(cmd, shell=True, cwd=os.path.join(os.curdir, "aws_layer"))