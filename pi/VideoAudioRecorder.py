import cv2
import threading
import assemblyai as aai
from dotenv import load_dotenv
import os
import requests
from flask import Flask, jsonify
import time
import pyaudio

# Load environment variables
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_APIKEY")

FLASK_SERVER_URL = 'http://10.37.118.10:5000'  # computer server URL


class VideoAudioRecorder:
    def __init__(self):
        self.recording = True
        self.start_time = 0
        self.sample_rate = 16_000
        self.transcriber = None
        self.recorder = VideoAudioRecorder()

    # Initialize Flask server for receiving transcriptions
    app = Flask(__name__)

    def start_video_recording(self):
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

        print("Recording video...")

        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video recording stopped.")

    def start_audio_transcription(self):
        def on_open(session_opened: aai.RealtimeSessionOpened):
            print("Session ID:", session_opened.session_id)
            self.start_time = time.time()

        def on_data(transcript: aai.RealtimeTranscript):
            if not transcript.text:
                return

            if isinstance(transcript, aai.RealtimeFinalTranscript):
                print(transcript.text, end="\r\n")
                # Send transcription to Flask server
                try:
                    response = requests.post(
                        FLASK_SERVER_URL + "/transcribe",
                        json={'text': transcript.text, 'time': time.time() - self.start_time}
                    )
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

        self.transcriber = aai.RealtimeTranscriber(
            sample_rate=self.sample_rate,
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close,
            end_utterance_silence_threshold=5000
        )

        self.transcriber.connect()

        print("Transcribing audio...")

        microphone_stream = aai.extras.MicrophoneStream(sample_rate=self.sample_rate)
        self.transcriber.stream(microphone_stream)

    def stop(self):
        self.recording = False
        if self.transcriber:
            self.transcriber.close()

    @app.route('/start-recording', methods=['GET'])
    def start_recording():
        # Start video recording in a new thread
        video_thread = threading.Thread(target=recorder.start_video_recording)
        video_thread.start()

        # Start audio transcription in a new thread
        audio_thread = threading.Thread(target=recorder.start_audio_transcription)
        audio_thread.start()

        return jsonify({"message": "Recording started"}), 200

    @app.route('/stop-recording', methods=['GET'])
    def stop_recording():
        global recorder
        
        recorder.stop()
        return jsonify({"message": "Recording stopped"}), 200


if __name__ == '__main__':
    VideoAudioRecorder.app.run(debug=True, host='0.0.0.0', port=6000)
