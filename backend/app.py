from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os


def create_app():
    load_dotenv()

    MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_DB = os.getenv("MONGO_DB", "collection")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0")

    app = Flask(__name__)

    # Construct MongoDB Atlas URI
    mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.ejp2g.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # Set up MongoDB client
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[MONGO_DB]
    items = db['items']

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
