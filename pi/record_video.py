import cv2
import threading

# Initialize the webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define the codec and create a VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# Flag to control the recording loop
recording = True

def stop_recording():
    global recording
    input("Press Enter to stop recording...\n")
    recording = False

# Start a thread to wait for user input to stop recording
thread = threading.Thread(target=stop_recording)
thread.start()

print("Recording...")

while recording:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Write the frame into the file 'output.mp4'
    out.write(frame)

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
print("Recording stopped.")