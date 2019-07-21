import json
import os
import sys

sys.path.append("aws_layer/python") # must precede layer-requiring imports
import src.api_v1 as api_v1

def lambda_handler(event, context):
    return api_v1.main(event)