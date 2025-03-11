from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def image_to_array(image_path):
    """Loads an image, converts it to RGB, and normalizes it."""
    img = Image.open(image_path).convert('RGB')  # Ensure it's in RGB format
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Example image paths (update these with your actual dataset)
image_paths = [
    'dog-vs-cat/cat/cat1.jpg', 'dog-vs-cat/cat/cat2.jpg',
    'dog-vs-cat/cat/cat3.jpg', 'dog-vs-cat/cat/cat4.jpg', 'dog-vs-cat/cat/cat5.jpg'
]

# Load images and labels
image_data = np.array([image_to_array(path) for path in image_paths])
labels = np.array([0, 1, 0, 1, 0])  # Example: 0 = Cat, 1 = Dog

# Convert labels to categorical (for classification)
labels = to_categorical(labels, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)

# Check data shapes
print("X_train shape:", X_train.shape)  # Expected: (num_samples, 64, 64, 3)
print("y_train shape:", y_train.shape)  # Expected: (num_samples, 2)

# Define a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # Output layer (2 classes: cat vs. dog)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_test, y_test))

# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

