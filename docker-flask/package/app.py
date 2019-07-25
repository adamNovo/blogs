from flask import Flask

app = Flask(__name__)

@app.teardown_request
def teardown_request(exception=None):
    pass

@app.route("/")
def hello():
    return "Health check. No auth."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)