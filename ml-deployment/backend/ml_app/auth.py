import configparser
import os
from functools import wraps
from flask import request, Response

CONFIG = configparser.ConfigParser()
CONFIG.read(os.path.abspath(os.path.join(os.path.dirname( __file__ ), 
    "instance", "secrets.ini")))
x_api_key = CONFIG["DEV"]["X_API_KEY"]

def check_auth(api_key):
    return x_api_key == api_key

def not_authenticated():
    return Response("Invalid credentials.", 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("x-api-key")
        if not key or not check_auth(key):
            return not_authenticated()
        else:
            return f(*args, **kwargs)
    return decorated