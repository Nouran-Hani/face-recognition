import os
import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

# Parameters
image_dir = 'CV/Face-Detection/data/gray'  # change this to your dataset folder
image_size = (100, 100)  # resize all images to same size
n_components = 100  # number of PCA components

# Step 1: Load images and extract labels
image_paths = glob.glob(os.path.join(image_dir, '*.pgm'))  # or '*.ppm' if needed
images = []
labels = []

for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load {path}")
        continue
    img = cv2.resize(img, image_size)
    images.append(img.flatten())
    
    # Extract label from filename
    base = os.path.basename(path)
    name = "_".join(base.split('_')[:2])  # e.g., 'Aaron_Eckhart'
    labels.append(name)

images = np.array(images)
labels = np.array(labels)

print(f"Loaded {len(images)} images with shape {images.shape}")

# Count occurrences
label_counts = Counter(labels)
valid_labels = {label for label, count in label_counts.items() if count >= 2}

# Filter images and labels
filtered_images = []
filtered_labels = []
for img, label in zip(images, labels):
    if label in valid_labels:
        filtered_images.append(img)
        filtered_labels.append(label)

images = np.array(filtered_images)
labels = np.array(filtered_labels)

print(f"Filtered to {len(images)} images with >= 2 per class")

# Step 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Step 3: Apply PCA
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"PCA transformed shape: {X_train_pca.shape}")

# Step 4: Train SVM
clf = SVC(kernel='linear', class_weight='balanced')
clf.fit(X_train_pca, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test_pca)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
