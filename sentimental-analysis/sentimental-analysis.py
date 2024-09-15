import cv2
import numpy as np
import math
import time
from keras.models import load_model
from collections import Counter

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

if __name__ == "__main__":
    # emotion_dict = {
    #     0: 'neutral', 
    #     1: 'happiness', 
    #     2: 'surprise', 
    #     3: 'sadness',
    #     4: 'anger', 
    #     5: 'disgust', 
    #     6: 'fear'
    # }
 
    # cap = cv2.VideoCapture('video1.mp4')
    cap = cv2.VideoCapture(0)
 
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    # result = cv2.VideoWriter('result.avi', 
    #                      cv2.VideoWriter_fourcc(*'MJPG'),
    #                      10, size)
 
    # # Read ONNX model
    # model = 'onnx_model.onnx'
    # model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
     
    # Read the Caffe face detector.
    model_path = 'RFB-320/RFB-320.caffemodel'
    proto_path = 'RFB-320/RFB-320.prototxt'
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    emotion_history = []
    start_time = time.time()
 
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
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
                emotion_prediction = emotion_model.predict(face_reshaped)
                max_index = np.argmax(emotion_prediction[0])
                emotion_label = emotion_dict[max_index]
                confidence = round(emotion_prediction[0][max_index] * 100)

                # Add the detected emotion to history
                emotion_history.append(emotion_label)

                # Draw a bounding box around the face and label it with the detected emotion
                cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_ori, emotion_label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                cv2.putText(img_ori, str(confidence) + "%", (x1, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Calculate the elapsed time
            elapsed_time = time.time() - start_time

            # Check if 5 seconds have passed
            if elapsed_time >= 5:
                # Determine the most common emotion in the last 5 seconds
                if emotion_history:
                    most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
                    cv2.putText(img_ori, f"5s Sentiment: {most_common_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # Reset the history and timer
                emotion_history = []
                start_time = time.time()
            
            # Display the output frame with bounding boxes and emotion labels
            cv2.imshow("Emotion Detector", img_ori)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
