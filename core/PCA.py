import os
import cv2
import numpy as np
import glob
import joblib 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from core.faceDetection import face_detection
from collections import Counter
from sklearn.model_selection import train_test_split
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier


def safe_pca(data, n_components):
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Handle small sample case properly
    if centered.shape[0] < centered.shape[1]:
        print("[PCA] Using dual covariance trick (n_samples < n_features)...")
        # Compute XXᵀ instead of XᵀX
        cov = np.dot(centered, centered.T) / centered.shape[0]
        eigs, eigvecs = np.linalg.eigh(cov)

        # Convert to XᵀX eigenvectors (may contain zeros)
        eigvecs = np.dot(centered.T, eigvecs)

        # Filter out zero vectors and normalize
        valid_indices = []
        for i in range(eigvecs.shape[1]):
            norm = np.linalg.norm(eigvecs[:, i])
            if norm > 1e-10:
                eigvecs[:, i] /= norm
                valid_indices.append(i)

        # Keep only non-zero eigenvectors
        eigvecs = eigvecs[:, valid_indices]
        eigs = eigs[valid_indices]
    else:
        print("[PCA] Using regular covariance...")
        cov = np.dot(centered.T, centered) / centered.shape[0]
        eigs, eigvecs = np.linalg.eigh(cov)


    # Sort descending
    print("[PCA] Sorting eigenvalues and eigenvectors...")
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    eigvecs = eigvecs[:, idx]

    print("[PCA] PCA complete.")
    return eigvecs[:, :n_components].T, eigs[:n_components], mean


class EigenFaceRecognition:
    def __init__(self, image_dir=None, image_size=(100, 100)):
        self.image_dir = image_dir
        self.image_size = image_size
        self.classifier = None
        self.projected_images = None
        self.labels = None
        self.eigvecs = None
        self.mean = None
        self.le = LabelEncoder()
        self.is_trained = False  

    def load_images_and_labels(self):
        image_size = (100, 100)  # resize all images to same size
        n_components = 100 
        print("[Load] Gathering image paths...")
        image_paths = glob.glob(os.path.join(self.image_dir, '*', '*.jpg'))
        if not image_paths:
            print(f"[Load] No images found in directory: {self.image_dir}")
            return np.array([]), np.array([])

        images = []
        labels = []

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load {path}")
                continue
            img = self.normalize_image(img) 
            img = cv2.resize(img, image_size)
            images.append(img.flatten())
            
            # Extract label from filename
            base = os.path.basename(path)
            name = "_".join(base.split('_')[:2])  # e.g., 'Aaron_Eckhart'
            labels.append(name)

            # data augmentation
            for aug in self.augment_image(img):
                images.append(aug.flatten())
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

        self.le.fit(labels)

        return images, labels
    
    def normalize_image(self, img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def augment_image(self, img):
        aug_images = []
        # Horizontal flip
        flipped = cv2.flip(img, 1)
        flipped_resized = cv2.resize(flipped, self.image_size)  # Resize to (100, 100)
        aug_images.append(flipped_resized)

        # Slight rotation
        M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle=10, scale=1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        rotated_resized = cv2.resize(rotated, self.image_size)  # Resize to (100, 100)
        aug_images.append(rotated_resized)

        # Brightness increase
        bright = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
        bright_resized = cv2.resize(bright, self.image_size)  # Resize to (100, 100)
        aug_images.append(bright_resized)

        # Scaling (zooming)
        scaled = cv2.resize(img, None, fx=1.1, fy=1.1)
        scaled_resized = cv2.resize(scaled, self.image_size)  # Resize to (100, 100)
        aug_images.append(scaled_resized)

        return aug_images


    def train_classifier(self):
        # Load the images and labels
        images, labels = self.load_images_and_labels()

        if images.size == 0:
            print("[Train] No images to train the classifier.")
            return

        # Perform PCA on the loaded images
        print("[Train] Performing PCA...")
        eigvecs, eigvals, mean = safe_pca(images, n_components=150)

        # Project the images into the PCA space
        X_train_pca = np.dot(images - mean, eigvecs.T)

        
        # Train the SVM classifier
        print("[Train] Training the SVM classifier...")
        clf = SVC(kernel='linear', C=1, class_weight='balanced')
        clf.fit(X_train_pca,  self.le.transform(labels))
        scores = cross_val_score(clf, X_train_pca, labels, cv=5)

        y_pred = cross_val_predict(clf, X_train_pca, labels, cv=5)
        print("[Train] Classification Report:")
        print(classification_report(labels, y_pred))
        print(f"[Train] Cross-validated accuracy: {np.mean(scores):.2f}")


        # clf.fit(X_train, y_train)

        # print("[Train] Training the k-NN classifier...")
        # clf = KNeighborsClassifier(n_neighbors=8)
        # clf.fit(X_train, y_train)
        
        # Store the classifier, PCA components, and mean
        self.classifier = clf
        self.eigvecs = eigvecs
        self.mean = mean
        self.is_trained = True

        print("[Train] Training complete.")

        self.save_model()
    
    def recognize_face(self, test_image):
        if not self.is_trained:
            print("[Recognize] Classifier has not been trained.")
            return None

        print("[Recognize] Preprocessing test image...")

        # Detect faces in the test image
        faces = face_detection(test_image)
        if not faces:
            print("[Recognize] No face detected in the image.")
            return None

        # Use the first detected face (you can improve this by selecting among multiple faces if needed)
        x, y, w, h = faces[0]
        face_crop = test_image[y:y+h, x:x+w]

        # Resize and flatten the face
        img_resized = cv2.resize(face_crop, self.image_size)
        img_resized = self.normalize_image(img_resized)
        img_flat = img_resized.flatten()

        # Project the image into the PCA space
        img_projected = np.dot(img_flat - self.mean, self.eigvecs.T)

        img_projected = img_projected.reshape(1, -1)

        # Predict the label of the image using the trained classifier
        print("[Recognize] Predicting label...")
        prediction = self.classifier.predict(img_projected)
        label = self.le.inverse_transform(prediction)

        print("[Recognize] Prediction complete.")
        return label[0]

    def save_model(self):
        dir = 'CV/face-recognition/core'
        path = os.path.join(dir, "model.pkl")  # Ensure it's a file path

        joblib.dump({
            'classifier': self.classifier,
            'eigvecs': self.eigvecs,
            'mean': self.mean,
            'label_encoder': self.le
        }, path)
        print(f"[Model] Model saved to: {path}")

    def load_model(self):
        dir = ''
        path = os.path.join(dir, "model.pkl")
        path = "core/model.pkl"

        if not os.path.exists(path):
            print(f"[Model] Model file not found in: {path}")
            return

        data = joblib.load(path)
        self.classifier = data['classifier']
        self.eigvecs = data['eigvecs']
        self.mean = data['mean']
        self.le = data['label_encoder']
        self.is_trained = True
        print(f"[Model] Model loaded from: {path}")

    def predict_from_image_path(self, image):
        if not self.is_trained:
            print("[Predict] Model not loaded or trained.")
            return None

        test_image = image
        if test_image is None:
            print(f"[Predict] Failed to load image: {image_path}")
            return None

        print("[Predict] Recognizing face from image path...")
        return self.recognize_face(test_image)

# if __name__ == "__main__":
#     dataset_path = r"CV/face-recognition/subjects"
#     test_image_path = r"CV/face-recognition/subjects/Jodie_Foster/Jodie_Foster_0001.jpg"

#     print("[Main] Initializing EigenFaceRecognition...")
#     eigen_face = EigenFaceRecognition(image_dir=dataset_path, image_size=(100, 100))

#     print("[Main] Training classifier...")
#     eigen_face.train_classifier()

#     print("[Main] Loading test image...")
#     test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
#     if test_image is None:
#         print(f"[Main] Failed to load test image: {test_image_path}")
#     else:
#         print("[Main] Recognizing face...")
#         predicted_label = eigen_face.recognize_face(test_image)
#         if predicted_label:
#             print(f"[Main] Predicted label: {predicted_label}")

#             color_image = cv2.imread(test_image_path)
#             cv2.putText(color_image, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.imshow("Predicted Face", color_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
