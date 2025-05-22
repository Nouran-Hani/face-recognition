from core.PCA import EigenFaceRecognition

def run_model(image):
    eigen_face = EigenFaceRecognition(image_dir="../subjects")
    eigen_face.load_model()

    label = eigen_face.predict_from_image_path(image)
    print("Predicted:", label)
    return label