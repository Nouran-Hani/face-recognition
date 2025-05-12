import os
import cv2
import numpy as np
import glob
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from faceDetection import face_detection

# Safe PCA function to handle edge cases and ensure it's computed once
def safe_pca(data, n_components):
    print("[PCA] Starting PCA...")
    data = np.array(data)
    if len(data.shape) != 2:
        raise ValueError(f"Data should be a 2D array, but got shape {data.shape}")

    print("[PCA] Calculating mean and centering data...")
    mean = np.mean(data, axis=0)
    centered = data - mean

    if centered.shape[0] < centered.shape[1]:
        print("[PCA] Using dual covariance trick (n_samples < n_features)...")
        cov = np.dot(centered, centered.T) / centered.shape[0]
        eigs, eigvecs = np.linalg.eigh(cov)
        eigvecs = np.dot(centered.T, eigvecs)
        valid_indices = []
        for i in range(eigvecs.shape[1]):
            norm = np.linalg.norm(eigvecs[:, i])
            if norm > 1e-10:
                eigvecs[:, i] /= norm
                valid_indices.append(i)
        eigvecs = eigvecs[:, valid_indices]
        eigs = eigs[valid_indices]
    else:
        print("[PCA] Using regular covariance...")
        cov = np.dot(centered.T, centered) / centered.shape[0]
        eigs, eigvecs = np.linalg.eigh(cov)

    print("[PCA] Sorting eigenvalues and eigenvectors...")
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    eigvecs = eigvecs[:, idx]

    print("[PCA] PCA complete.")
    return eigs[:n_components], eigvecs[:, :n_components], mean


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
        self.is_trained = False  # Track if the model has been trained

    def load_images_and_labels(self):
        print("[Load] Gathering image paths...")
        image_paths = glob.glob(os.path.join(self.image_dir, '*', '*.jpg'))
        if not image_paths:
            print(f"[Load] No images found in directory: {self.image_dir}")
            return np.array([]), np.array([])

        images = []
        labels = []

        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[Load] Failed to load {path}")
                continue
            img = cv2.resize(img, self.image_size)
            images.append(img.flatten())
            label = os.path.basename(os.path.dirname(path))
            labels.append(label)

        images = np.array(images)
        labels = np.array(labels)

        if images.size == 0:
            print("[Load] No valid images loaded.")
            return np.array([]), np.array([])

        print(f"[Load] Loaded {len(images)} images with {len(np.unique(labels))} unique labels.")
        return images, labels

    def train_classifier(self):
        if self.is_trained:
            print("[Train] Classifier is already trained. Skipping training.")
            return

        print("[Train] Starting training process...")
        images, labels = self.load_images_and_labels()

        if images.size == 0:
            print("[Train] No images to train the classifier.")
            return

        print("[Train] Encoding labels...")
        labels_encoded = self.le.fit_transform(labels)

        print("[Train] Performing PCA...")
        eigvals, eigvecs, mean = safe_pca(images, n_components=50)

        print("[Train] Projecting training images into eigenspace...")
        self.eigvecs = eigvecs
        self.mean = mean
        self.projected_images = np.dot(images - mean, eigvecs)

        print("[Train] Training SVM classifier...")
        self.classifier = SVC(kernel='rbf', probability=True)
        self.classifier.fit(self.projected_images, labels_encoded)

        self.is_trained = True  # Mark model as trained
        print("[Train] Training complete.")

    def recognize_face(self, test_image):
        if not self.is_trained:
            print("[Recognize] Classifier has not been trained.")
            return None

        print("[Recognize] Preprocessing test image...")

        faces = face_detection(test_image)
        if not faces:
            print("[Recognize] No face detected in the image.")
            return None

        # Use the first detected face (you can enhance this to select among multiple faces if necessary)
        x, y, w, h = faces[0]
        face_crop = test_image[y:y+h, x:x+w]

        img_resized = cv2.resize(face_crop, self.image_size)
        img_flat = img_resized.flatten()
        img_projected = np.dot(img_flat - self.mean, self.eigvecs)
        img_projected = img_projected.reshape(1, -1)

        print("[Recognize] Predicting label...")
        prediction = self.classifier.predict(img_projected)
        label = self.le.inverse_transform(prediction)
        print("[Recognize] Prediction complete.")
        return label[0]

if __name__ == "__main__":
    dataset_path = r"data_uncropped"
    test_image_path = r"data_uncropped/Aaron_Guiel/Aaron_Guiel_0001.jpg"

    print("[Main] Initializing EigenFaceRecognition...")
    eigen_face = EigenFaceRecognition(image_dir=dataset_path, image_size=(100, 100))

    print("[Main] Training classifier...")
    eigen_face.train_classifier()

    print("[Main] Loading test image...")
    test_image = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    if test_image is None:
        print(f"[Main] Failed to load test image: {test_image_path}")
    else:
        print("[Main] Recognizing face...")
        predicted_label = eigen_face.recognize_face(test_image)
        if predicted_label:
            print(f"[Main] Predicted label: {predicted_label}")

            color_image = cv2.imread(test_image_path)
            cv2.putText(color_image, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Predicted Face", color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
