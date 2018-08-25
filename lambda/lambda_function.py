import json
import pandas as pd
import numpy as np

def lambda_handler(event, context):
    mode = event["queryStringParameters"]["mode"]
    print(np.zeros((5,5)))
    print(pd.DataFrame(columns=["col1", "col2"]))
    return response({"message": "API mode: {}".format(mode)}, 200)

def response(message, status_code):
    return {
        'statusCode': str(status_code),
        'body': json.dumps(message),
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
            },
        }

if __name__ == "__main__":
    event = {
        "queryStringParameters": {
            "mode": "mode"
        }
    }
    res = lambda_handler(event=event, context={})
    print(res)