from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os


def create_app():
    load_dotenv()

    MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_DB = os.getenv("MONGO_DB", "your_database_name")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0.ejp2g")

    app = Flask(__name__)

    # Construct MongoDB Atlas URI
    mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # Set up MongoDB client
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[MONGO_DB]

    @app.route('/items', methods=['GET'])
    def get_items():
        collection = db['items']
        items = list(collection.find({}, {'_id': 0}))
        return jsonify(items)

    @app.route('/items', methods=['POST'])
    def add_item():
        collection = db['items']
        item = request.json
        result = collection.insert_one(item)
        return jsonify({"message": "Item added successfully", "id": str(result.inserted_id)}), 201

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
