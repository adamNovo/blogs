import argparse
import os
import subprocess
import sys

container = "lambda_api" # use container_name from docker-compose.yml

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--action", required=True, choices=["start", "stop", "rm", "rebuild", "ssh"], help="select action")
args = vars(ap.parse_args())
action = args["action"]

def main():
    if action == "start":
        docker_start()
    elif action == "stop":
        docker_stop()
    elif action == "rm":
        docker_rm()
    elif action == "rebuild":
        docker_rebuild()
    elif action == "ssh":
        docker_ssh()
    
def docker_ssh():
    subprocess.call("docker exec -it {} /bin/bash".format(container), shell=True)

def docker_start():
    subprocess.call("""
    docker-compose -f docker-compose.yml up;""", shell=True)

def docker_stop():
    subprocess.call("""
    docker stop {0};
    """.format(container), shell=True)

def docker_rm():
    subprocess.call("""
    docker stop {0};
    docker rm {0};
    """.format(container), shell=True)

def docker_rebuild():
    subprocess.call("""
    docker-compose -f docker-compose.yml up -d --force-recreate --build""", shell=True)

if __name__ == "__main__":
    main()