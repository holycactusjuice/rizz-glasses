import assemblyai as aai
from dotenv import load_dotenv
import os
import requests  # Import requests library
from flask import Flask, request, jsonify

app = Flask(__name__)

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_APIKEY")

FLASK_SERVER_URL = 'http://172.20.10.9:5000'  # Flask server URL


def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)


def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end="\r\n")
        # Send transcription to Flask server
        try:
            response = requests.post(
                FLASK_SERVER_URL + "/transcribe", json={'text': transcript.text})
            response_data = response.json()
            print(f"Server Response: {response_data}")
        except Exception as e:
            print(f"Error sending data to server: {e}")
    else:
        print(transcript.text, end="\r")


def on_error(error: aai.RealtimeError):
    print("An error occurred:", error)


def on_close():
    print("Closing Session")

transcriber = aai.RealtimeTranscriber(
    sample_rate=16_000,
    on_data=on_data,
    on_error=on_error,
    on_open=on_open,
    on_close=on_close,
    end_utterance_silence_threshold=5000
)

@app.route('/start-recording', methods={'GET'})
def start_recording():
    transcriber.connect()

    print("Recording started")

    microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
    transcriber.stream(microphone_stream)

@app.route('/stop-recording', methods={'GET'})
def stop_recording():
    transcriber.close()

    return "Recording stopped"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)