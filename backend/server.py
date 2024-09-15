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

    @app.route('/', methods={'GET'})
    def hello():
        return "Hello, Flask Server!\n"

    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        try:
            data = request.get_json()  # Expecting JSON data
            # Extract 'text' from JSON payload
            transcription_text = data.get('text', '')
            print(f"Received transcription: {transcription_text}")
            # You can add additional processing here if needed
            response = {"status": "success",
                        "received_text": transcription_text}
        except Exception as e:
            print(f"Error: {e}")
            response = {"status": "error", "message": str(e)}

        return jsonify(response)


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)  # Adjust the port as needed
