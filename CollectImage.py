import cv2
import mediapipe as mp
import os
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Variables
image_count = 0
max_images = 100
collecting = False
waiting_for_next_class = False
current_class = "default_class"
save_path = "hand_datasets"
message = ""
running = True  # Controls the OpenCV loop


# Create the save directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)


# OpenCV capture loop
def capture_hand_images():
    global image_count, collecting, waiting_for_next_class, message, running

    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Draw hand landmarks and collect images
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if collecting and image_count < max_images:
                    # Extract bounding box
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

                    # Adjust bounding box
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max, y_max = min(w, x_max), min(h, y_max)

                    # Crop hand region
                    hand_crop = frame[y_min:y_max, x_min:x_max]

                    # Save image
                    class_path = os.path.join(save_path, current_class)
                    os.makedirs(class_path, exist_ok=True)
                    image_path = os.path.join(class_path, f"{image_count}.jpg")
                    cv2.imwrite(image_path, hand_crop)
                    image_count += 1

        # Add message overlay
        if message:
            cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Convert frame to ImageTk format and display in Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        if image_count >= max_images and not waiting_for_next_class:
            collecting = False
            waiting_for_next_class = True
            message = f"Collected {max_images} images for class '{current_class}'. Enter next class name and press Start again."

        if not running:
            break

    cap.release()


# Tkinter functions
def start_collecting():
    global collecting, message, waiting_for_next_class
    if waiting_for_next_class:
        messagebox.showinfo("Info", "Please enter the next class name and press Start.")
        return
    collecting = True
    message = f"Collecting images for class '{current_class}'."


def stop_collecting():
    global collecting, message
    collecting = False
    message = "Stopped collecting images."


def exit_program():
    global running
    running = False
    root.destroy()


def set_class_name():
    global current_class, image_count, waiting_for_next_class, message
    new_class = class_name_entry.get().strip()
    if not new_class:
        messagebox.showwarning("Warning", "Class name cannot be empty!")
        return
    current_class = new_class
    image_count = 0
    waiting_for_next_class = False
    message = f"Switched to class '{current_class}'."


# Tkinter GUI
root = tk.Tk()
root.title("HomeSync Dataset Collection")
icon = tk.PhotoImage(file='./resources/icon.png')
root.iconphoto(False, icon)
root.minsize(648, 600)
root.resizable(False, False)

# add the app background
background = tk.PhotoImage(file="./resources/background.png")
tk.Label(root, image=background, border=0).place(x=0, y=0)

# Video display label
video_label = tk.Label(root)
video_label.place(x=0, y=1)

# Class name entry
tk.Label(root, text="Enter Class Name:", font=("Tahoma", 16)).place(x=0, y=500)
class_name_entry = tk.Entry(root, width=30, font=("Tahoma", 16))
class_name_entry.place(x=190, y=500)

set_class_button = tk.Button(root, text="Set Class", command=set_class_name, font=("Tahoma", 12), width=10)
set_class_button.place(x=538, y=500)

# Buttons for start, stop, and exit
start_button = tk.Button(root, text="Start Collection", command=start_collecting, bg="green", fg="white", font=("Tahoma", 12))
start_button.place(x=0, y=550)

stop_button = tk.Button(root, text="Stop Collection", command=stop_collecting, bg="orange", fg="white", font=("Tahoma", 12))
stop_button.place(x=150, y=550)

exit_button = tk.Button(root, text="Exit Program", command=exit_program, bg="red", fg="white", font=("Tahoma", 12))
exit_button.place(x=250, y=550)

# Start OpenCV in a separate thread
threading.Thread(target=capture_hand_images, daemon=True).start()

root.mainloop()
