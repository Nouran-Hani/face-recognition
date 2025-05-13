from PCA import EigenFaceRecognition

eigen_face = EigenFaceRecognition(image_dir="CV/face-recognition/subjects")
eigen_face.load_model()

label = eigen_face.predict_from_image_path("CV/face-recognition/final data/Jodie_Foster_0001.jpg")
print("Predicted:", label)