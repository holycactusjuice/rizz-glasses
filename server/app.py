from flask import Flask, jsonify, request
from pymongo import MongoClient
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
from llmservice import GroqConversationAnalyzer

# Global variable to set the context
CONVERSATION_CONTEXT = "date"  # Can be changed to "business" as needed


def create_app():
    load_dotenv()

    MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_DB = os.getenv("MONGO_DB", "collection")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    analyzer = GroqConversationAnalyzer(GROQ_API_KEY)

    app = Flask(__name__)

    # Construct MongoDB Atlas URI
    mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.ejp2g.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # Set up MongoDB client
    mongo_client = MongoClient(mongo_uri, tlsInsecure=True)
    db = mongo_client[MONGO_DB]
    items = db['items']

    app.config['MONGO_URI'] = mongo_uri
    mongo = PyMongo(app)

    def insert_new_transcription(unix_timestamp):
        items.insert_one({
            "unix_timestamp": unix_timestamp,
            "frequency_table": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
                "6": 0,
                "7": 0,
                "8": 0,
                "9": 0,
                "10": 0
            },
            "total_time": 0,
            "transcription": [],
            "summary": ""
        })

        print("New transcription inserted")

        return

    def update_transcription(unix_timestamp, transcription_data):
        query = {"unix_timestamp": unix_timestamp}  # get from pi

        items.update_one(query, {
            # update frequency table
            "$inc": {f"score_table.{transcription_data['score']}": 1},
            # update total time
            "$set": {
                "total_time": transcription_data['time'],
                # update transcription
            },
            "$push": {
                f"transcription": {
                    "time": transcription_data['time'],
                    "dialogue": transcription_data['dialogue'],
                    "sentiment": transcription_data['sentiment'],
                    "explanation": transcription_data['explanation'],
                    "score": transcription_data['score']
                }
            },
        })

        print("Transcription updated")

        return

    def update_summary(unix_timestamp, summary):
        query = {"unix_timestamp": unix_timestamp}
        items.update_one(query, {
            "$set": {"summary": summary}
        })
        print("Summary updated")

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    @app.route('/stop-recording', methods=['POST'])
    def stop_recording():
        response = request.get_json()
        unix_timestamp = response.get('start_recording_timestamp')
        update_summary(unix_timestamp, analyzer.summarize_conversation())
        return jsonify({"message": "Recording stopped"})

    @app.route('/transcribe', methods=['POST'])
    def handle_transcription():
        global CONVERSATION_CONTEXT
        transcription_json = request.get_json()

        conversation_timestamp = transcription_json['start_recording_timestamp']
        dialogue = transcription_json['text']
        transcription_timestamp = transcription_json['time']
        sentiment = transcription_json['sentiment']

        # Use the global context variable
        if CONVERSATION_CONTEXT == "date":
            score, explanation = analyzer.score_sentence(dialogue)
            suggestion = analyzer.return_suggestion()
        elif CONVERSATION_CONTEXT == "business":
            score, explanation = analyzer.score_sentence(
                dialogue, type="business")
            suggestion = analyzer.return_suggestion(type="business")
        else:
            return jsonify({"error": "Invalid conversation context"}), 400

        data = {
            "dialogue": dialogue,
            "sentiment": sentiment,
            "explanation": explanation,
            "score": score,
            "time": transcription_timestamp
        }

        print("data:", data, "suggestion:", suggestion)

        conversation_query = {
            "unix_timestamp": conversation_timestamp}  # get from pi

        if items.find_one(conversation_query) is None:
            insert_new_transcription(conversation_timestamp)
        update_transcription(conversation_timestamp, data)
        return jsonify({"suggestion": suggestion})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
