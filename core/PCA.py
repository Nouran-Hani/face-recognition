import numpy as np
import cv2


def safe_pca(data, n_components):
    mean = np.mean(data, axis=0)
    centered = data - mean

    # Handle small sample case properly
    if centered.shape[0] < centered.shape[1]:
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
        cov = np.dot(centered.T, centered) / centered.shape[0]
        eigs, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    eigvecs = eigvecs[:, idx]

    return eigvecs[:, :n_components].T, eigs[:n_components], mean


if __name__ == "__main__":
    # Load and preprocess two images
    img1 = cv2.imread("../data/gray/Alecos_Markides_0001.pgm", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../data/gray/Alec_Baldwin_0004.pgm", cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error loading images")
        exit()

    # Convert to 50x50 vectors
    size = (50, 50)
    vec1 = cv2.resize(img1, size).flatten().astype(np.float32) / 255.0
    vec2 = cv2.resize(img2, size).flatten().astype(np.float32) / 255.0

    # Compute PCA (will automatically handle the 2-sample case)
    components, _, mean = safe_pca(np.vstack([vec1, vec2]), n_components=1)

    # Project and compare
    proj1 = np.dot(vec1 - mean, components.T)
    proj2 = np.dot(vec2 - mean, components.T)

    distance = np.linalg.norm(proj1 - proj2)
    threshold = 12.0  # Empirical value for 50x50 images

    print(f"Distance: {distance:.2f}")
    print("Same person" if distance < threshold else "Different people")

    # Visualization
    cv2.imshow("Image 1", cv2.resize(img1, (200, 200)))
    cv2.imshow("Image 2", cv2.resize(img2, (200, 200)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()