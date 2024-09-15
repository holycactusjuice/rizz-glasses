from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/', methods={'GET'})
def hello():
    return "Hello, Flask Server!\n"


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()  # Expecting JSON data
        # Extract 'text' from JSON payload
        transcription_text = data.get('text', '')
        sentiment = data.get('sentiment', '')
        start_recording_timestamp = data.get('start_recording_timestamp', '')
        print(f"Received transcription: {transcription_text}")
        # You can add additional processing here if needed
        response = {"status": "success", "sentiment": sentiment, "received_text": transcription_text, "start_recording_timestamp": start_recording_timestamp}
    except Exception as e:
        print(f"Error: {e}")
        response = {"status": "error", "message": str(e)}

    return jsonify(response)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    try:
        data = request.get_json()  # Expecting JSON data
        # Extract 'text' from JSON payload
        sentiment = data.get('text', '')
        print(f"Received sentiment: {sentiment}")
        # You can add additional processing here if needed
        response = {"status": "success", "received_sentiment": sentiment}
    except Exception as e:
        print(f"Error: {e}")
        response = {"status": "error", "message": str(e)}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)  # Adjust the port as needed
