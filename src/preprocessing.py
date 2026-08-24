from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os

IMG_SIZE = 32
DATASET_PATH = "dataset/GTSRB/Training"

images = []
labels = []

for class_id in range(43):
    class_path = os.path.join(DATASET_PATH, f"{class_id:05d}")

    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is not None:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            images.append(image)
            labels.append(class_id)

images = np.array(images)
labels = np.array(labels)

images = images / 255.0

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
print("Dataset loaded successfully!")

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Training labels:", y_train.shape)
print("Validation labels:", y_val.shape)

os.makedirs("processed_data", exist_ok=True)

np.save("processed_data/X_train.npy", X_train)
np.save("processed_data/X_val.npy", X_val)
np.save("processed_data/y_train.npy", y_train)
np.save("processed_data/y_val.npy", y_val)

print("Processed data saved successfully!")