from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from llmservice import GroqConversationAnalyzer

def create_app():
    load_dotenv()

    MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_DB = os.getenv("MONGO_DB", "your_database_name")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    analyzer = GroqConversationAnalyzer(GROQ_API_KEY)

    app = Flask(__name__)

    # # Construct MongoDB Atlas URI
    # mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # # Set up MongoDB client
    # mongo_client = MongoClient(mongo_uri)
    # db = mongo_client[MONGO_DB]

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    @app.route('/transcribe', methods=['POST'])
    def recieve_transcription():
        transcription_json = request.json()
        print(transcription_json)
        return jsonify({"message": "Transcription received"})
    
    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
