import configparser
import datetime
import json
import os
import pytest
import time
import sys

sys.path.append("/mnt/app/") # must precede app-requiring imports
sys.path.append("/mnt/app/aws_layer/python") # must precede layer-requiring imports used in Lambda layer

import lambda_function

os.environ["lambda_stage"] = "test-invoke-stage"
os.environ["run_lambda"] = "True"

@pytest.fixture
def secrets_fix():
    CONFIG = configparser.ConfigParser()
    CONFIG.read(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
        "..", "src", "instance", "secrets.ini")))
    api_key = CONFIG[os.environ["lambda_stage"]]["api_key"]
    lambda_url = CONFIG[os.environ["lambda_stage"]]["lambda_url"]
    return api_key, lambda_url

# @pytest.mark.skip()
def test_local_home():
    event = {
        "queryStringParameters": None,
        "path": "/api",
        "requestContext": {
            "stage": os.environ["lambda_stage"]
        }
    }
    res = lambda_function.lambda_handler(event=event, context={})
    assert 200 == int(res["statusCode"])
    res_body = json.loads(res["body"])
    assert res_body["msg"] == "path /api OK"

@pytest.mark.skipif(os.environ["run_lambda"] != "True", reason="Skipped")
def test_lambda_home(secrets_fix):
    import requests
    api_key, lambda_url = secrets_fix
    headers = {"x-api-key": api_key}
    r = requests.get(lambda_url, headers=headers)
    # print(r.json()["event"])
    assert r.status_code == 200

