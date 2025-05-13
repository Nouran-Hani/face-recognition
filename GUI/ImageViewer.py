import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QFileDialog, QVBoxLayout, QGroupBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal, QPoint


class ImageViewer(QWidget):
    imageChanged = pyqtSignal(object)
    selectionMade = pyqtSignal(tuple)  # Emits ((x1, y1), (x2, y2))

    def __init__(self,title = "image"):
        super().__init__()

        self.initializeUI(title)
        self.setupLayout()
        self.styleUI()
        self.setReadOnly(False)

        self.start_point = None
        self.end_point = None
        self.selecting = False
        self.image_with_rect = None
        self.red_point = None  # Track the current red point
        self.remove_previous_dot = True  # Flag to control dot removal

    def initializeUI(self, title):
        self.groupBox = QGroupBox(title)

        self.image = None
        self.isReadOnly = True
        self.image_label = QLabel("Double-click to upload an image", self)
        self.image_label.setAlignment(Qt.AlignCenter)

    def setupLayout(self):
        layout = QVBoxLayout()
        self.groupBox.setLayout(layout)  # Set the QGroupBox layout
        layout.addWidget(self.image_label)  # Add QLabel inside QGroupBox

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.groupBox)  # Add QGroupBox to the main layout
        self.setLayout(main_layout)

    def styleUI(self):
        """Applies styling to the QGroupBox"""
        self.groupBox.setStyleSheet("""
            QGroupBox {
                background-color: #f7f7f7;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid #007BFF;
                border-radius: 10px;
                margin-top: 15px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 5px;
                color: #007BFF;
                border-radius: 5px;
            }
        """)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and not self.isReadOnly:
            self.openImage()

    def openImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp)")
        if file_path:
            self.displayImage(cv2.imread(file_path))

    def displayImage(self, image):
        """Displays an image in the QLabel."""
        if image is not None:
            self.image = image
            self.image_with_rect = image.copy()  # Keep a copy for selection
            self.imageChanged.emit(self.image)

            self.updatePixmap()

    def updatePixmap(self):
        """Updates QLabel with the current image (with or without selection box)."""
        if self.image_with_rect is not None:
            image_rgb = cv2.cvtColor(self.image_with_rect, cv2.COLOR_BGR2RGB)
            height, width, channels = image_rgb.shape
            bytes_per_line = channels * width
            q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(
                self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # def mousePressEvent(self, event):
    #     """Handles mouse click and emits (x, y) pixel coordinates."""
    #     if self.image is not None and event.button() == Qt.LeftButton:
    #         widget_pos = event.pos()
    #         image_pos = self.widgetToImageCoords(widget_pos)
    #
    #         if image_pos is not None:
    #             self.selectionMade.emit(image_pos)
    #             print(image_pos)



    #
    # def mouseMoveEvent(self, event):
    #     """Handles mouse movement and updates selection rectangle."""
    #     if self.selecting and self.image is not None:
    #         self.end_point = event.pos()
    #         self.drawRectangle()
    #
    # def mouseReleaseEvent(self, event):
    #     """Handles mouse release and finalizes selection."""
    #     if self.selecting and self.image is not None:
    #         self.selecting = False
    #         self.end_point = event.pos()
    #
    #         # Convert widget coordinates to image coordinates
    #         start = self.widgetToImageCoords(self.start_point)
    #         end = self.widgetToImageCoords(self.end_point)
    #
    #         if start and end:
    #             self.selectionMade.emit((start, end))  # Emit the selection
    #             print(f"Selected Region: Start {start}, End {end}")
    def mousePressEvent(self, event):
        """Handles mouse click, removes previous point based on flag, and adds a new one."""
        if self.image is not None and event.button() == Qt.LeftButton:
            widget_pos = event.pos()
            image_pos = self.widgetToImageCoords(widget_pos)

            if image_pos is not None:
                # Emit the position (you can still keep this if you need it for other purposes)
                self.selectionMade.emit(image_pos)
                print(image_pos)

                # If the flag is True and there is a previous red point, remove it
                if self.remove_previous_dot and self.red_point is not None:
                    self.red_point = None
                    self.update()  # Clear the previous dot from the image

                # Draw the new red point
                self.red_point = image_pos
                self.draw_on_image(np.array([image_pos]), color=Qt.red, thickness=6)

    def draw_on_image(self, coordinates, color=Qt.green, thickness=4, close_shape=True):
        """
        General function to draw points on the image.

        :param coordinates: NumPy array of shape (N, 2) representing points.
        :param color: Color of the points (default: red).
        :param thickness: Size of the points (default: 4).
        :param close_shape: Not used in this case but kept for consistency.
        """
        if coordinates is None or coordinates.shape[0] < 1:  # Check for valid input
            print("Error: Not enough points to draw")
            return

        if self.pixmap is None:
            print("Error: No image loaded")
            return

        # Create a copy of the pixmap to draw on
        pixmap_copy = self.pixmap.copy()
        painter = QPainter(pixmap_copy)
        pen = QPen(color, thickness)
        painter.setPen(pen)

        # Draw a point (small circle) for each coordinate
        for i in range(coordinates.shape[0]):  # Loop through all coordinates
            x, y = coordinates[i]
            # Draw a small circle at the point (thickness will define the size)
            painter.drawEllipse(int(x) - thickness // 2, int(y) - thickness // 2, thickness, thickness)

        painter.end()

        # Set the updated image
        self.image_label.setPixmap(pixmap_copy)
    #
    # def drawRectangle(self):
    #     """Draws a rectangle on the image to show selection."""
    #     if self.image is not None and self.start_point and self.end_point:
    #         self.image_with_rect = self.image.copy()
    #
    #         start = self.widgetToImageCoords(self.start_point)
    #         end = self.widgetToImageCoords(self.end_point)
    #
    #         if start and end:
    #             cv2.rectangle(self.image_with_rect, start, end, (0, 0, 255), 2)
    #
    #         self.updatePixmap()
    def widgetToImageCoords(self, widget_point):
        """Converts QLabel widget coordinates to actual image coordinates."""
        if self.image is None or self.image_label.pixmap() is None:
            return None

        # Get original image dimensions
        img_height, img_width = self.image.shape[:2]

        # Get the current displayed pixmap
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return None

        # Get the geometry of the image label within the widget
        label_rect = self.image_label.geometry()

        # Calculate the position of the pixmap within the label (centered)
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()

        x_offset = (label_rect.width() - pixmap_width) // 2
        y_offset = (label_rect.height() - pixmap_height) // 2

        # Convert widget coordinates to label coordinates
        label_point = self.image_label.mapFromParent(widget_point)

        # Adjust to pixmap coordinates
        pixmap_x = label_point.x() - x_offset
        pixmap_y = label_point.y() - y_offset

        # Check if the point is within the pixmap
        if (pixmap_x < 0 or pixmap_y < 0 or
                pixmap_x >= pixmap_width or pixmap_y >= pixmap_height):
            return None

        # Calculate the scaling factors
        scale_x = img_width / pixmap_width
        scale_y = img_height / pixmap_height

        # Convert to image coordinates
        img_x = int(pixmap_x * scale_x)
        img_y = int(pixmap_y * scale_y)

        # Ensure coordinates are within image bounds
        img_x = max(0, min(img_x, img_width - 1))
        img_y = max(0, min(img_y, img_height - 1))

        return img_x, img_y
    # def getSelection(self):
    #     """Returns the selected region's start and end points."""
    #     if self.start_point and self.end_point:
    #         return self.widgetToImageCoords(self.start_point), self.widgetToImageCoords(self.end_point)
    #     return None

    def setReadOnly(self, enabled: bool):
        """Enables or disables image uploading."""
        self.isReadOnly = enabled
        if enabled:
            self.image_label.setText("Image viewing mode only")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    #
    # def on_selection_made(coords):
    #     print(f"Selected Region Coordinates: {coords}")
    #     print(viewer.image.shape[1])  # Width
    #     print(viewer.image.shape[0])  # Height


    # viewer.selectionMade.connect(on_selection_made)

    viewer.show()
    sys.exit(app.exec_())
