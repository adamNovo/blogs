from app import app as application

def flask_default():
    application.run(debug=True, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    flask_default()