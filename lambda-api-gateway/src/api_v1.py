import configparser
import json
import pandas as pd
import requests
import os

from src import utils

def main(event):
    """
    Params:
        event: {
            "resource": "/api",
            "path": "/api",
            "httpMethod": "GET",
            "headers": null,
            "multiValueHeaders": null,
            "queryStringParameters": {
                "cf": "1,2,3",
                "math": "dcf"
            },
            "multiValueQueryStringParameters": {
                "cf": [
                    "1,2,3"
                ],
                "math": [
                    "dcf"
                ]
            },
            "pathParameters": null,
            "stageVariables": null,
            "requestContext": {
            "path": "/api",
            "accountId": "503499648497",
            "resourceId": "3js4yj",
            "stage": "test-invoke-stage",
            "domainPrefix": "testPrefix",
            "requestId": "db18d801-2b7f-11e9-9d32-358785cc0223",
            "identity": {
                "cognitoIdentityPoolId": null,
                "cognitoIdentityId": null,
                "apiKey": "test-invoke-api-key",
                "cognitoAuthenticationType": null,
                "userArn": "arn:aws:iam::503499648497:root",
                "apiKeyId": "test-invoke-api-key-id",
                "userAgent": "aws-internal/3 aws-sdk-java/1.11.481 Linux/4.9.137-0.1.ac.218.74.329.metal1.x86_64 OpenJDK_64-Bit_Server_VM/25.192-b12 java/1.8.0_192",
                "accountId": "503499648497",
                "caller": "503499648497",
                "sourceIp": "test-invoke-source-ip",
                "accessKey": "ASIAXKOXK5XY2MV6AMPR",
                "cognitoAuthenticationProvider": null,
                "user": "503499648497"
            },
            "domainName": "testPrefix.testDomainName",
            "resourcePath": "/api",
            "httpMethod": "GET",
            "extendedRequestId": "Uxe2kHrZIAMF5_g=",
            "apiId": "qycwdw29o8"
            },
            "body": null,
            "isBase64Encoded": false
        }
    """
    os.environ["lambda_stage"] = event["requestContext"]["stage"]
    if event["queryStringParameters"] == None:
        # default path
        return utils.response({"msg": "path /api OK"}, 200)
    elif "math" in event["queryStringParameters"] and event["queryStringParameters"]["math"] == "dcf":
        pv = calc_dcf(event)
        return utils.response({"pv": pv.to_json(orient="split")}, 200)
    else:
        return utils.response({"msg": "Path fail"}, 400)

