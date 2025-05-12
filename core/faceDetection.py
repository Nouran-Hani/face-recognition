import cv2
import numpy as np

def face_detection(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Skin color detection (for color images)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Combine skin and edge information
    combined = cv2.bitwise_and(skin_mask, cv2.bitwise_not(edges))
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    processed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    faces = []
    for contour in contours:
        # Filter by area
        area = cv2.contourArea(contour)
        if area < 2000:  
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio 
        aspect_ratio = w / h
        if not (0.6 < aspect_ratio < 1.8):
            continue
            
        # check for eye regions
        roi = gray[y:y+h, x:x+w]
        if not validate_face_region(roi):
            continue
            
        faces.append((x, y, w, h))
    
    return faces

def validate_face_region(face_roi):
   
    eyes = detect_eyes(face_roi)
    return len(eyes) >= 1  # At least one eye detected

def detect_eyes(face_roi):
   
    # Equalize histogram
    equalized = cv2.equalizeHist(face_roi)
    
    # Threshold to find dark regions (eyes)
    _, thresh = cv2.threshold(equalized, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    eyes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 500:  # Reasonable eye area range
            eyes.append(cnt)
    
    return eyes

def main():
    
    test_image_path = "data_uncropped/Abba_Eban/Abba_Eban_0001.jpg"
    
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error: Could not load image from {test_image_path}")
        return
    
    if test_image.shape[1] > 1000:
        test_image = cv2.resize(test_image, (0,0), fx=0.5, fy=0.5)
    
    faces = face_detection(test_image)
    print(f"Detected {len(faces)} faces")
    
    result_image = test_image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    combined = np.hstack((test_image, result_image))
    cv2.imshow("Original vs Detection", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()