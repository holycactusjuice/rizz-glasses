import assemblyai as aai
from dotenv import load_dotenv
import os
import requests  # Import requests library
import time

load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_APIKEY")

FLASK_SERVER_URL = 'http://172.20.10.9:5000'  # Flask server URL

start_time = 0


def on_open(session_opened: aai.RealtimeSessionOpened):
    print("Session ID:", session_opened.session_id)
    start_time = time.time()


def on_data(transcript: aai.RealtimeTranscript):
    if not transcript.text:
        return

    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(transcript.text, end="\r\n")
        # Send transcription to Flask server
        try:
            response = requests.post(
                FLASK_SERVER_URL + "/transcribe", json={'text': transcript.text, 'time': time.time() - start_time})
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

transcriber.connect()

microphone_stream = aai.extras.MicrophoneStream(sample_rate=16_000)
transcriber.stream(microphone_stream)

transcriber.close()
