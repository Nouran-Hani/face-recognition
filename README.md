<h1 align="center">
    <img alt="Face Detection and Recognition Demo" src="Readme/demo.gif" />
</h1>

<h1 align="center">Face Detection and Recognition</h1>
<div align="center">
  <img src="https://github.com/user-attachments/assets/86dabe7f-beeb-450d-8846-0acaf89c5336" >
</div>

<h4 align="center"> 
	Status: ✅ Completed
</h4>

<p align="center">
 <a href="#about">About</a> •
 <a href="#features">Features</a> •
 <a href="#tech-stack">Tech Stack</a> •  
 <a href="#developers">Developers</a>
</p>

---

## 🧠 About

The **Face Detection and Recognition** project implements a complete pipeline for detecting human faces in images and recognizing individuals using advanced computer vision and machine learning techniques. The system combines manual face detection methods with PCA-based feature extraction and SVM classification for robust performance.

This tool serves as a foundational platform for tasks in:
- Biometric authentication
- Surveillance systems
- Photo organization
- Human-computer interaction

---

## ✨ Features

### 👁️ Face Detection Pipeline
- **Skin Color Segmentation** in HSV color space
- **Canny Edge Detection** for feature enhancement
- **Morphological Filtering** for noise reduction
- **Geometric Analysis** for face-like region validation
- **Eye Detection** for final verification


### 📊 Dimensionality Reduction
- **Principal Component Analysis (PCA)**
- Eigenface extraction
- Adaptive covariance computation
- Adjustable component count for performance tuning

### 🤖 Machine Learning
- **Support Vector Machine (SVM)** classification
- 5-fold cross-validation
- Detailed classification reports
- Model persistence for reuse

### 📈 Performance Evaluation
- **ROC curve analysis**
- One-vs-Rest multi-class strategy
- True Positive Rate vs False Positive Rate visualization
- AUC metric calculation



### 🔄 Workflow Integration
- Image preprocessing (histogram equalization)
- Data augmentation (flipping, rotation, scaling)
- Face detection module
- Recognition interface

---

## ⚙️ Tech Stack

- **Python**
- **OpenCV**
- **NumPy**
- **SciPy**
- **scikit-learn**
- **Matplotlib**

---

## 👨‍💻 Developers

| [**Talal Emara**](https://github.com/TalalEmara) | [**Meram Mahmoud**](https://github.com/Meram-Mahmoud) | [**Maya Mohammed**](https://github.com/Mayamohamed207) | [**Nouran Hani**](https://github.com/Nouran-Hani) |
|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|:------------------------------------------:|

---

## 📎 Learn More

* [Face Detection Techniques](https://en.wikipedia.org/wiki/Face_detection)
* [PCA for Face Recognition](https://en.wikipedia.org/wiki/Eigenface)
* [SVM Classification](https://scikit-learn.org/stable/modules/svm.html)
* [ROC Curve Analysis](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
