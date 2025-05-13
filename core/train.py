from PCA import EigenFaceRecognition

img_dir = 'D:/Projects/DSP/CV/face-recognition/subjects'
pca = EigenFaceRecognition(img_dir)

clf = pca.train_classifier()


