import os
import subprocess
import sys

cmd = """
zip -r Archive.zip 
lambda_function.py 
src 
--exclude=*.DS_Store* 
--exclude=*__pycache__* 
--exclude=*.pyc* 
"""

cmd = "".join(cmd.splitlines()) # remove newlines
subprocess.call(cmd, shell=True)