import cv2
import numpy as np
import threading
import assemblyai as aai
from dotenv import load_dotenv
import os
import requests
from flask import Flask, jsonify, request
import time
import math
from keras.models import load_model
from collections import Counter
import pyaudio

# Load environment variables
load_dotenv()
aai.settings.api_key = os.getenv("ASSEMBLY_APIKEY")

FLASK_SERVER_URL = 'http://172.20.10.9:5000'  # computer server URL


class VideoAudioRecorder:
    def __init__(self):
        self.recording = True
        self.start_time = 0
        self.sample_rate = 16_000
        self.transcriber = None
        self.recorder = None
        self.emotion_history = {}

    def start_video_recording(self):
        image_mean = np.array([127, 127, 127])
        image_std = 128.0
        iou_threshold = 0.3
        center_variance = 0.1
        size_variance = 0.2
        min_boxes = [
            [10.0, 16.0, 24.0], 
            [32.0, 48.0], 
            [64.0, 96.0], 
            [128.0, 192.0, 256.0]
        ]
        strides = [8.0, 16.0, 32.0, 64.0]
        threshold = 0.5

        emotion_model = load_model('FER_model.h5')
        emotion_dict = {
            0: 'angry', 
            1: 'disgust', 
            2: 'fear', 
            3: 'happy', 
            4: 'neutral', 
            5: 'sad', 
            6: 'surprise'
        }
        def define_img_size(image_size):
            shrinkage_list = []
            feature_map_w_h_list = []
            for size in image_size:
                feature_map = [int(math.ceil(size / stride)) for stride in strides]
                feature_map_w_h_list.append(feature_map)
        
            for i in range(0, len(image_size)):
                shrinkage_list.append(strides)
            priors = generate_priors(
                feature_map_w_h_list, shrinkage_list, image_size, min_boxes
            )
            return priors

        def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
            priors = []
            for index in range(0, len(feature_map_list[0])):
                scale_w = image_size[0] / shrinkage_list[0][index]
                scale_h = image_size[1] / shrinkage_list[1][index]
                for j in range(0, feature_map_list[1][index]):
                    for i in range(0, feature_map_list[0][index]):
                        x_center = (i + 0.5) / scale_w
                        y_center = (j + 0.5) / scale_h
        
                        for min_box in min_boxes[index]:
                            w = min_box / image_size[0]
                            h = min_box / image_size[1]
                            priors.append([
                                x_center,
                                y_center,
                                w,
                                h
                            ])
            return np.clip(priors, 0.0, 1.0)

        def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
            scores = box_scores[:, -1]
            boxes = box_scores[:, :-1]
            picked = []
            indexes = np.argsort(scores)
            indexes = indexes[-candidate_size:]
            while len(indexes) > 0:
                current = indexes[-1]
                picked.append(current)
                if 0 < top_k == len(picked) or len(indexes) == 1:
                    break
                current_box = boxes[current, :]
                indexes = indexes[:-1]
                rest_boxes = boxes[indexes, :]
                iou = iou_of(
                    rest_boxes,
                    np.expand_dims(current_box, axis=0),
                )
                indexes = indexes[iou <= iou_threshold]
            return box_scores[picked, :]
        
        def area_of(left_top, right_bottom):
            hw = np.clip(right_bottom - left_top, 0.0, None)
            return hw[..., 0] * hw[..., 1]
        
        
        def iou_of(boxes0, boxes1, eps=1e-5):
            overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
            overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:]) 
        
            overlap_area = area_of(overlap_left_top, overlap_right_bottom)
            area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
            area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
            return overlap_area / (area0 + area1 - overlap_area + eps)

        def predict(
            width, 
            height, 
            confidences, 
            boxes, 
            prob_threshold, 
            iou_threshold=0.3, 
            top_k=-1
        ):
            boxes = boxes[0]
            confidences = confidences[0]
            picked_box_probs = []
            picked_labels = []
            for class_index in range(1, confidences.shape[1]):
                probs = confidences[:, class_index]
                mask = probs > prob_threshold
                probs = probs[mask]
                if probs.shape[0] == 0:
                    continue
                subset_boxes = boxes[mask, :]
                box_probs = np.concatenate(
                    [subset_boxes, probs.reshape(-1, 1)], axis=1
                )
                box_probs = hard_nms(box_probs,
                                    iou_threshold=iou_threshold,
                                    top_k=top_k,
                                    )
                picked_box_probs.append(box_probs)
                picked_labels.extend([class_index] * box_probs.shape[0])
            if not picked_box_probs:
                return np.array([]), np.array([]), np.array([])
            picked_box_probs = np.concatenate(picked_box_probs)
            picked_box_probs[:, 0] *= width
            picked_box_probs[:, 1] *= height
            picked_box_probs[:, 2] *= width
            picked_box_probs[:, 3] *= height
            return (
                picked_box_probs[:, :4].astype(np.int32), 
                np.array(picked_labels), 
                picked_box_probs[:, 4]
            )
        
        def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
            if len(priors.shape) + 1 == len(locations.shape):
                priors = np.expand_dims(priors, 0)
            return np.concatenate([
                locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
                np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
            ], axis=len(locations.shape) - 1)
        
        def center_form_to_corner_form(locations):
            return np.concatenate(
                [locations[..., :2] - locations[..., 2:] / 2,
                locations[..., :2] + locations[..., 2:] / 2], 
                len(locations.shape) - 1
            )


        cap = cv2.VideoCapture(-1)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

        print("Recording video...")
        
        # Read the Caffe face detector.
        model_path = 'RFB-320/RFB-320.caffemodel'
        proto_path = 'RFB-320/RFB-320.prototxt'
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        input_size = [320, 240]
        width = input_size[0]
        height = input_size[1]
        priors = define_img_size(input_size)

        while self.recording:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            else:
                out.write(frame)
                frame = cv2.flip(frame, 1)
                img_ori = frame
                # Resize and prepare the frame for face detection
                rect = cv2.resize(img_ori, (width, height))
                rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
                net.setInput(cv2.dnn.blobFromImage(
                    rect, 1 / image_std, (width, height), 127)
                )

                # Perform face detection using the existing network
                boxes, scores = net.forward(["boxes", "scores"])
                boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
                scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
                boxes = convert_locations_to_boxes(
                    boxes, priors, center_variance, size_variance
                )
                boxes = center_form_to_corner_form(boxes)

                # Perform prediction and get face bounding boxes
                picked_boxes, picked_labels, picked_scores = predict(
                    frame_width, frame_height, scores, boxes, threshold
                )

                # Loop over detected faces
                for i, box in enumerate(picked_boxes):
                    x1, y1, x2, y2 = box

                    # Extract the face region from the frame
                    face = img_ori[y1:y2, x1:x2]

                    # Preprocess the face for the emotion model
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_resized = cv2.resize(face_gray, (48, 48))  # Resize to match model input size
                    face_normalized = face_resized / 255.0
                    face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))

                    # Run emotion prediction
                    emotion_prediction = emotion_model.predict(face_reshaped, verbose=0)
                    max_index = np.argmax(emotion_prediction[0])
                    emotion_label = emotion_dict[max_index]
                    confidence = round(emotion_prediction[0][max_index] * 100)

                    # Add the detected emotion to history
                    self.emotion_history.append(emotion_label, confidence)

                    # Draw a bounding box around the face and label it with the detected emotion
                    cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_ori, emotion_label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.putText(img_ori, str(confidence) + "%", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # # Check if 5 seconds have passed
                # if elapsed_time >= 5:
                #     # Determine the most common emotion in the last 5 seconds
                #     if emotion_history:
                #         most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
                #         cv2.putText(img_ori, f"5s Sentiment: {most_common_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #     # Reset the history and timer
                #     emotion_history = []
                #     start_time = time.time()
                #     # Send sentiment to Flask server
                #     try:
                #         response = requests.post(
                #             FLASK_SERVER_URL + "/sentiment",
                #             json={'text': most_common_emotion, 'time': time.time() - self.start_time}
                #         )
                #         response_data = response.json()
                #         print(f"Server Response: {response_data}")
                #     except Exception as e:
                #         print(f"Error sending data to server: {e}")
                
                # Display the output frame with bounding boxes and emotion labels (only if display is present)
                # cv2.imshow("Emotion Detector", img_ori)

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
                # Process the most common sentiment
                most_common_emotion = Counter(self.emotion_history.keys()).most_common(1)[0][0]
                emotion_count = 0
                total_confidence = 0
                for emotion in self.emotion_history.keys():
                    if emotion == most_common_emotion:
                        emotion_count += 1
                        total_confidence += self.emotion_history[emotion]

                avg_confidence = total_confidence // emotion_count

                self.emotion_history.clear()
                # Send transcription to Flask server
                try:
                    response = requests.post(
                        FLASK_SERVER_URL + "/transcribe",
                        json={'text': transcript.text, 'sentiment': most_common_emotion, 'confidence': avg_confidence, 'context': context, 'start_recording_timestamp': start_recording_timestamp, 'time': time.time() - self.start_time}
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

recorder = VideoAudioRecorder()
# Initialize Flask app
app = Flask(__name__)
start_recording_timestamp = 0
context = None

@app.route('/start-recording', methods=['POST'])
def start_recording():
    global start_recording_timestamp
    start_recording_timestamp = time.time()

    # get the context
    context = request.args.get('context')

    # Start video recording in a new thread
    video_thread = threading.Thread(target=recorder.start_video_recording)
    video_thread.start()

    # Start audio transcription in a new thread
    audio_thread = threading.Thread(target=recorder.start_audio_transcription)
    audio_thread.start()

    return jsonify({"message": "Recording started"}), 200

@app.route('/stop-recording', methods=['GET'])
def stop_recording():
    recorder.stop()
    requests.post(FLASK_SERVER_URL + "/stop-recording", 
                  json={'start_recording_timestamp': start_recording_timestamp}
                  )
    return jsonify({"message": "Recording stopped"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=6000)
