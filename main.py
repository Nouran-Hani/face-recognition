import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, \
    QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QBoxLayout, QCheckBox
from GUI.ImageViewer import ImageViewer
from core.test import run_model

from GUI.styles import GroupBoxStyle, button_style

import time
from PIL import Image

from core.PCA import EigenFaceRecognition
from core.faceDetection import draw_detected_faces


class FetchFeature(QMainWindow):
    def __init__(self):
        super().__init__()  # Initialize QMainWindow
        self.setWindowTitle("Raqib")
        self.setFixedSize(1200, 800)

        self.initializeUI()
        self.createRecogniseParameters()
        self.setupLayout()
        self.styleUI()
        self.connectUI()

        self.eigen_face = EigenFaceRecognition()

    def initializeUI(self):

        self.processingImage = None
        self.currentMode = "Face Detection"
        self.logo = QLabel("Catch")

        def createModePanel():
            self.detectButton = QPushButton("Face Detection")
            self.recogniseButton = QPushButton("Face Recognision")

            self.detectButton.clicked.connect(lambda: self.changeMode("Face Detection"))
            self.recogniseButton.clicked.connect(lambda: self.changeMode("Face Recognision"))


        createModePanel()

        self.inputViewer = ImageViewer("Input Image")
        self.outputViewer = ImageViewer("Output Image")
        self.outputViewer.setReadOnly(True)
        self.secondInputViewer = ImageViewer("Input Image")

        self.processButton = QPushButton("Process")




        

    def createRecogniseParameters(self):
        self.parametersGroupBox = QGroupBox("Face Recognision")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        self.topEigensLabel = QLabel("Top Eigne values:")
        self.topEigensLabel.setAlignment(Qt.AlignCenter)

        self.topEigensSpinBox = QSpinBox()

        self.resultLabel = QLabel("Not in dataset")
        self.resultLabel.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        # layout.addWidget(self.topEigensLabel)
        # layout.addWidget(self.topEigensSpinBox)
        layout.addWidget(self.resultLabel)


        self.parametersGroupBox.setLayout(layout)



    def createDetectParameters(self):
        self.parametersGroupBox = QGroupBox("Face Detection")
        self.parametersGroupBox.setStyleSheet(GroupBoxStyle)

        # self.topEigensLabel = QLabel("Top Eigne values:")
        # self.topEigensLabel.setAlignment(Qt.AlignCenter)
        #
        # self.topEigensSpinBox = QSpinBox()

        self.resultLabel = QLabel("")
        self.resultLabel.setAlignment(Qt.AlignCenter)

        layout = QHBoxLayout()
        # layout.addWidget(self.topEigensLabel)
        # layout.addWidget(self.topEigensSpinBox)
        layout.addWidget(self.resultLabel)


        self.parametersGroupBox.setLayout(layout)





    def setupLayout(self):
        mainWidget = QWidget(self)
        self.setCentralWidget(mainWidget)

        mainLayout = QHBoxLayout()
        modesLayout = QVBoxLayout()
        workspace = QVBoxLayout()
        self.imagesLayout = QHBoxLayout()
        imagesLayoutH = QHBoxLayout()
        self.parametersLayout = QHBoxLayout()

        self.parametersLayout.addWidget(self.parametersGroupBox)
        self.parametersLayout.addWidget(self.processButton)

        # Add widgets to layout
        modesLayout.addWidget(self.logo, alignment=Qt.AlignCenter)

        modesLayout.addWidget(self.detectButton)
        modesLayout.addWidget(self.recogniseButton)
        # modesLayout.addWidget(self.spectralButton)
        # modesLayout.addWidget(self.localButton)

        # modesLayout.addWidget(self.regionButton)
        # modesLayout.addWidget(self.kmeansButton)
        # modesLayout.addWidget(self.meanShiftButton)
        # modesLayout.addWidget(self.agglomerativeButton)
        # modesLayout.addWidget(self.houghCirclesButton)
        # modesLayout.addWidget(self.houghEllipseButton)
        # modesLayout.addWidget(self.snakeButton)
        modesLayout.addStretch()



        self.imagesLayout.addWidget(self.inputViewer)
        self.imagesLayout.addWidget(self.outputViewer)
        # Nest layouts
        mainLayout.addLayout(modesLayout,20)
        mainLayout.addLayout(workspace,80)

        workspace.addLayout(self.imagesLayout)
        workspace.addLayout(self.parametersLayout)


        mainWidget.setLayout(mainLayout)

    def changeMode(self, mode):
        """Change the current mode and update the UI accordingly."""
        self.currentMode = mode

        # Remove existing parametersGroupBox if it exists
        if hasattr(self, "parametersGroupBox"):
            self.parametersLayout.removeWidget(self.parametersGroupBox)
            self.parametersGroupBox.deleteLater()  # Properly delete the widget

        # Create the corresponding parameter panel
        if mode == "Face Detection":
            self.createDetectParameters()

        elif mode == "Face Recognision":
            self.createRecogniseParameters()


        self.parametersLayout.insertWidget(0, self.parametersGroupBox)

    def styleUI(self):
        self.logo.setStyleSheet("font-family: 'Franklin Gothic';"
                                " font-size: 32px;"
                                " font-weight:600;"
                                " padding:30px;")


        self.processButton.setFixedWidth(250)
        self.processButton.setFixedHeight(40)
        # self.processButton.setStyleSheet(second_button_style)
        self.detectButton.setStyleSheet(button_style)
        # self.regionButton.setStyleSheet(button_style)
        self.recogniseButton.setStyleSheet(button_style)
        # self.kmeansButton.setStyleSheet(button_style)
        # self.spectralButton.setStyleSheet(button_style)
        # self.localButton.setStyleSheet(button_style)
        # self.meanShiftButton.setStyleSheet(button_style)
        # self.agglomerativeButton.setStyleSheet(button_style)



    def connectUI(self):
        self.processButton.clicked.connect(self.processImage)
        self.inputViewer.selectionMade.connect(self.on_selection_made)

    def on_selection_made(self,coords):
        self.selected = coords

    def processImage(self):
        self.processingImage = self.inputViewer.image.copy()
        if self.currentMode == "Face Detection":
            self.processingImage = draw_detected_faces(self.processingImage)
        elif self.currentMode == "Face Recognision":
            # self.resultLabel.setText(self.eigen_face.recognize_face(self.processingImage))
            self.resultLabel.setText(run_model(self.processingImage))


        self.outputViewer.displayImage(self.processingImage)






if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = FetchFeature()
    window.show()
    sys.exit(app.exec_())
