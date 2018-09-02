import json
import subprocess

application = "ml-tutorial"
folder = "ml_app"

subprocess.call("""
heroku container:login;
heroku container:push web -R -a {0} --arg HTTPS=on;
heroku container:release web -a {0};
docker rmi registry.heroku.com/{0}/web;
""".format(application), shell=True)