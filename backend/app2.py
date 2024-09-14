from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from llmservice import GroqConversationAnalyzer

def create_app():
    # Load environment variables
    load_dotenv()

    api_key = os.environ.get("GROQ_API_KEY")

    # Initialize LLM
    analyzer = GroqConversationAnalyzer(api_key)

    # MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    # MONGO_USER = os.getenv("MONGO_USER")
    # MONGO_DB = os.getenv("MONGO_DB", "your_database_name")
    # MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0.ejp2g")

    app = Flask(__name__)

    # Construct MongoDB Atlas URI
    # mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # Set up MongoDB client
    # mongo_client = MongoClient(mongo_uri)
    # db = mongo_client[MONGO_DB]

    # @app.route('/items', methods=['GET'])
    # def get_items():
    #     # collection = db['items']
    #     # items = list(collection.find({}, {'_id': 0}))
    #     # return jsonify(items)
    #     return jsonify({"message": "Get items endpoint"})

    # @app.route('/items', methods=['POST'])
    # def add_item():
    #     # collection = db['items']
    #     # item = request.json
    #     # result = collection.insert_one(item)
    #     # return jsonify({"message": "Item added successfully", "id": str(result.inserted_id)}), 201
    #     return jsonify({"message": "Add item endpoint"}), 201

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    @app.route('/suggest', methods=['GET', 'POST'])
    def get_suggestion():
        suggestion = analyzer.return_suggestion()
        return jsonify({"suggestion": suggestion})

    @app.route('/analyze', methods=['GET', 'POST'])
    def analyze_input():
        input_text = request.json.get('text', '')
        results = analyzer.process_input(input_text)
        return jsonify(results)

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)