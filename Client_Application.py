import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import subprocess
import socket


# Connect to Raspberry Pi server
server_ip = '192.168.0.196'  # Raspberry Pi's IP address
server_port = 8888
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port)) 

np.set_printoptions(suppress=True)

# Load the model
model = load_model("./Trained_Models/keras_Model.h5", compile=False)

# Load the labels
with open("./Trained_Models/labels.txt", "r") as f:
    class_names = f.readlines()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize camera
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

def process_frame():
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image.")
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    prediction_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            hand_image = frame[y_min:y_max, x_min:x_max]

            if hand_image.size == 0:
                continue

            hand_image_resized = cv2.resize(hand_image, (224, 224), interpolation=cv2.INTER_AREA)
            hand_image_array = np.asarray(hand_image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
            hand_image_array = (hand_image_array / 127.5) - 1

            prediction = model.predict(hand_image_array)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()
            confidence_score = prediction[0][index]

            prediction_text = f"{class_name}: {confidence_score * 100:.2f}%"

                # Check if confidence level is 80%
            if confidence_score >= 0.7:
                # Send message via socket connection including the class name
                message = f"{class_name[2:]}"
                client_socket.send(message.encode())

            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    prediction_label.config(text=prediction_text)
    lmain.after(10, process_frame)

# Set up GUI
root = tk.Tk()
root.title("HomeSync Dashboard")
# Load the icon image
icon = tk.PhotoImage(file='./resources/icon.png')

def run_script():
    #release the camera
    camera.release()
    #close the hands detections
    hands.close()
    cv2.destroyAllWindows()
    #close the current window
    root.destroy()
    # Runs the other .py file
    subprocess.run(["python", "DatasetMaker.py"])

# Set the window icon and size
root.iconphoto(False, icon)
root.minsize(965,600)
root.resizable(False,False)
#add the app background
background = tk.PhotoImage(file = "./resources/background.png")
bkg = tk.Label(root, image = background, border=0).place(x=0,y=0)
# Create a label to display the camera feed
lmain = tk.Label(root,border=0)
lmain.place(x=0,y=1)

# Create a label to display the predictions
prediction_label = tk.Label(root, text="No Hand Detected", font=("Helvetica", 16))
prediction_label.place(x=0,y=500)

#create a button to open the dataset maker
dataset_icon = tk.PhotoImage(file="./resources/create_dataset.png")
train_button=tk.Button(root, text = "Train Model",image=dataset_icon, border=0, command = run_script).place(x=650,y=10)

#create a button to open the training model
model_icon = tk.PhotoImage(file="./resources/train_model.png")
capture_b=tk.Button(root, text = "Train Model",image=model_icon, border=0, command = run_script).place(x=650,y=100)

# Start processing frames
process_frame()

# Run the Tkinter event loop
root.mainloop()

# Cleanup
camera.release()
hands.close()
cv2.destroyAllWindows()
