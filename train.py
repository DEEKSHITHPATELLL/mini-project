import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

dataset_dir = "E:/projects/miniproject/dataset/dataset/dataset"
img_height, img_width = 128, 128
batch_size = 32
num_epochs = 150
initial_lr = 0.001

gesture_matrix_file = "gesture_matrices.json"

def save_gesture_matrix(landmarks, gesture_name, file_path=gesture_matrix_file):
    normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]

    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        data = {}

    data[gesture_name] = normalized_landmarks

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Gesture '{gesture_name}' saved successfully!")

def predict_gesture(landmarks, file_path=gesture_matrix_file):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Gesture matrix file '{file_path}' not found.")
    with open(file_path, "r") as file:
        gesture_data = json.load(file)
    normalized_landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
    min_mse = float("inf")
    predicted_gesture = None

    for gesture_name, reference_landmarks in gesture_data.items():
        mse = np.mean([np.square(np.subtract(norm, ref)).sum() for norm, ref in zip(normalized_landmarks, reference_landmarks)])
        if mse < min_mse:
            min_mse = mse
            predicted_gesture = gesture_name

    return predicted_gesture

if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
print("Verifying dataset structure...")
for root, dirs, files in os.walk(dataset_dir):
    print(f"Directory: {root}, Subdirectories: {dirs}, Files: {len(files)}")
    if not dirs and len(files) == 0:
        print(f"Warning: Directory '{root}' is empty or does not contain valid images.")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)
val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)
if train_generator.samples == 0 or val_generator.samples == 0:
    raise ValueError("Training or validation set is empty. Please check your dataset.")

print(f"Number of training images: {train_generator.samples}")
print(f"Number of validation images: {val_generator.samples}")
num_classes = len(train_generator.class_indices)
print("Class labels:", train_generator.class_indices)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])
model.compile(
    optimizer=Adam(learning_rate=initial_lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
checkpoint = ModelCheckpoint(
    filepath="gesture_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=num_epochs,
    callbacks=[checkpoint, early_stopping],
    verbose=1
)
print("Evaluating the model...")
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
labels_path = "class_labels.txt"
with open(labels_path, "w") as f:
    for label in train_generator.class_indices.keys():
        f.write(f"{label}\n")
print(f"Class labels saved to '{labels_path}'.")
history_path = "training_history.npz"
np.savez(history_path, accuracy=history.history['accuracy'], val_accuracy=history.history['val_accuracy'], loss=history.history['loss'], val_loss=history.history['val_loss'])
print(f"Training history saved to '{history_path}'.")