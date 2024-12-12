import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

# Constants
dataset_path = "hand_datasets"  # Directory where the datasets are stored
image_size = (64, 64)  # Resize images to a uniform size
batch_size = 32
epochs = 10
model_output_path = "hand_model.h5"

# Load and preprocess data
def load_data(dataset_path, image_size):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))
    class_map = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name, class_idx in class_map.items():
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            image = cv2.imread(file_path)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            images.append(image)
            labels.append(class_idx)

    images = np.array(images, dtype="float32") / 255.0  # Normalize pixel values
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, labels, class_names

print("Loading data...")
images, labels, class_names = load_data(dataset_path, image_size)
print(f"Data loaded. Found {len(class_names)} classes.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
datagen.fit(X_train)

# Build CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

input_shape = (image_size[0], image_size[1], 3)
num_classes = len(class_names)
model = create_model(input_shape, num_classes)

# Train the model
print("Training the model...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_test, y_test),
    epochs=epochs,
    verbose=1
)

# Evaluate the model
print("Evaluating the model...")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save(model_output_path)
print(f"Model saved as {model_output_path}")

# Save class names for reference
class_map_path = "class_map.txt"
with open(class_map_path, "w") as f:
    for idx, class_name in enumerate(class_names):
        f.write(f"{idx}: {class_name}\n")
print(f"Class map saved as {class_map_path}")
