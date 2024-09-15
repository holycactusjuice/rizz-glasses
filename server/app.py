from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from llmservice import GroqConversationAnalyzer


def create_app():
    load_dotenv()

    MONGO_PASSWD = os.getenv("MONGO_PASSWD")
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_DB = os.getenv("MONGO_DB", "collection")
    MONGO_CLUSTER = os.getenv("MONGO_CLUSTER", "cluster0")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    time_curr = 0

    analyzer = GroqConversationAnalyzer(GROQ_API_KEY)

    app = Flask(__name__)

    # Construct MongoDB Atlas URI
    mongo_uri = f"mongodb+srv://{MONGO_USER}:{MONGO_PASSWD}@{MONGO_CLUSTER}.ejp2g.mongodb.net/{MONGO_DB}?retryWrites=true&w=majority"

    # Set up MongoDB client
    mongo_client = MongoClient(mongo_uri)
    db = mongo_client[MONGO_DB]
    items = db['items']

    def insert_new_transcription(unix_timestamp):
        items.insert_one({
            "unix_timestamp": unix_timestamp,
            "frequency_table": {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0
            },
            "total_time": 0,
            "transcription": {},
            "summary": ""
        })

        return jsonify({"message": "New transcription inserted"})

    def update_transcription(unix_timestamp, transcription_data):
        query = {"unix_timestamp": 0}  # get from pi

        items.update_one(query, {
            # update frequency table
            "$inc": {f"score_table.{transcription_data.score}": 1},
            # update total time
            "$set": {
                "total_time": transcription_data.time,
                # update transcription
                "$set": {f"transcription.{transcription_data.time}": {
                    "dialogue": transcription_data.dialogue,
                    "sentiment": transcription_data.sentiment,
                    "explanation": transcription_data.explanation,
                    "score": transcription_data.score
                }}
            },
        })

    def update_summary(unix_timestamp, summary):
        query = {"unix_timestamp": 0}
        items.update_one(query, {
            "$set": {"summary": summary}
        })

    @app.route('/')
    def hello():
        return "Hello, MongoDB Atlas!"

    @app.route('/stop-recording', methods={'GET'})
    def stop_recording():
        response = request.get_json()
        unix_timestamp = response.get('unix_timestamp')
        update_summary(unix_timestamp, analyzer.return_summary())
        return "Recording stopped"

    @app.route('/transcribe', methods=['POST', 'GET'])
    def handle_transcription():
        transcription_json = request.json()

        conversation_timestamp = transcription_json['start_recording_timestamp']
        dialogue = transcription_json['text']
        transcription_timestamp = transcription_json['time']
        sentiment = transcription_json['sentiment']

        analyzer.score_conversation(dialogue, terminal=False)
        suggestion = analyzer.return_suggestion()

        score, explanation = analyzer.score_sentence(
            dialogue, terminal=False)

        data = {
            "dialogue": dialogue,
            "sentiment": sentiment,
            "explanation": explanation,
            "score": score,
            "time": transcription_timestamp
        }

        conversation_query = {
            "unix_timestamp": conversation_timestamp}  # get from pi

        if items.find_one(conversation_query) is None:
            insert_new_transcription(
                conversation_timestamp, conversation_timestamp)

        update_transcription(conversation_timestamp, data)

        return jsonify({"suggestion": suggestion})

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=5000)
