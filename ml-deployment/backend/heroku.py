import json
import requests
import os
import subprocess
import sys
import generate_docker_files

application = "optimal-portfolio"
folder = "optimal_portfolio"
mode_options = ["devcloud", "stage", "prod"]
os_options = ["mac", "linux"]

def slack_log(msg):
    data = {"text": msg}
    slack_url = "https://hooks.slack.com/services/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"
    if os.environ["APP_MODE"] == "stage":
        slack_url = "https://hooks.slack.com/services/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"
    if os.environ["APP_MODE"] == "prod":
        slack_url = "https://hooks.slack.com/services/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"
    r = requests.post("{}".format(slack_url),
        data = json.dumps(data))

print("""
Generate docker files. App mode:
- options: {}
""".format(mode_options))
mode = input("Please make a selection: ")
if mode in mode_options:
    os.environ["APP_MODE"] = mode
    generate_docker_files.main(mode=mode)
    generate_docker_files.delete_dockerfile("Dockerfile.test")
else:
    sys.exit("Invalid mode")
os_input = input("Please enter os: ({})".format(os_options))
if os_input in os_options:
    if os_input == os_options[0]:
        sudo = ""
    elif os_input == os_options[1]:
        sudo = "sudo"
else:
    sys.exit("Invalid os")

print("""
Heroku action:
1 - container:push
2 - view free hours
""")
selection = input("Please make a selection: ")

if selection == "1":
    slack_log("Deploying {} APP to heroku".format(application))
    subprocess.call("""
    {2} heroku container:login;
    cd {3};
    {2} heroku container:push web -R -a {0}-{1} --arg HTTPS=on;
    {2} heroku container:release web -a {0}-{1};
    {2} docker rmi registry.heroku.com/{0}-{1}/web;
    """.format(application, mode, sudo, folder), shell=True)
    slack_log("Deployment {} APP completed".format(application))
elif selection == "2":
    subprocess.call("""
    {2} heroku ps -a {}-{};
    """.format(application, mode, sudo), shell=True)
    
else:
    sys.exit("Invalid selection.")