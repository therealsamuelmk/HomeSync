import os
import cv2
import mediapipe as mp

# Creating the path for the dataset collection
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 6
dataset_size = 100

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for the user to press 'Q' to start collecting data for this class
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.putText(frame, 'Press "Q" to start collecting data {}'.format(j), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Gesthub HomeSync', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:
        
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Calculate the bounding box
                    h, w, _ = image.shape
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        if x < x_min:
                            x_min = x
                        if x > x_max:
                            x_max = x
                        if y < y_min:
                            y_min = y
                        if y > y_max:
                            y_max = y

                    # Crop the hand region
                    hand_image = image[y_min:y_max, x_min:x_max]
                    if hand_image.size > 0:
                        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), hand_image)
                        counter += 1

            cv2.imshow('Gesthub HomeSync', image)

            # Add a delay to avoid capturing the same frame multiple times
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    print('Finished collecting data for class {}'.format(j))

cap.release()
cv2.destroyAllWindows()
