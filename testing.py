import cv2
import numpy as np
from joblib import load
import os
import random


detections = []

def getDetections():

    return detections


def select_random_image(image_directory):
   
   
    valid_extensions = ['.jpg', '.png']
    images = [file for file in os.listdir(image_directory) if any(file.endswith(ext) for ext in valid_extensions)]
    if not images:
        raise ValueError("No images found in the directory.")
    selected_image = random.choice(images)
    return os.path.join(image_directory, selected_image)

def read_annotations(annotation_path):
    boxes = []
    with open(annotation_path, 'rb') as file:  
        content = file.read().decode('utf-8', errors='ignore').replace('\0', '') 
    for line in content.splitlines():
        if line.strip():  
            parts = line.strip().split()  #
            if len(parts) >= 5:  
                try:
                    box = list(map(int, parts[1:5]))
                    boxes.append(box)
                except ValueError as e:
                    print(f"Skipping line with error: {e} - {line}")
    return boxes


def sliding_window(image, step_size, window_size=(64, 128)):
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_pedestrians_svm(model, image, scales):
    hog = cv2.HOGDescriptor()
    for scale in scales:
        scaled_image = cv2.resize(image, None, fx=scale, fy=scale)
        for (x, y, window) in sliding_window(scaled_image, step_size=4):  
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue
            features = hog.compute(window).reshape(1, -1)
            pred = model.predict(features)
            if pred == 1:
                score = model.decision_function(features)
                if score >= 0.9:
                    detections.append((x / scale, y / scale, score, scale))

    return non_max_suppression(detections, 0.1)

def detect_pedestrians_opencv(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, weights = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    return non_max_suppression_opencv(rects, 0.3)

def non_max_suppression(detections, threshold):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x[2], reverse=True)
    picked = []
    while detections:
        current = detections.pop(0)
        picked.append(current)
        detections = [det for det in detections if iou(current, det) < threshold]
    return picked

def non_max_suppression_opencv(rects, overlapThresh):
    if len(rects) == 0:
        return []
    pick = []
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = x1 + rects[:, 2]
    y2 = y1 + rects[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        for pos in range(0, last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = float(w * h) / area[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)
    return rects[pick]

def iou(det1, det2):
    x1, y1, score1, scale1 = det1
    x2, y2, score2, scale2 = det2
    w1, h1 = 64 * scale1, 128 * scale1
    w2, h2 = 64 * scale2, 128 * scale2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def draw_detections_combined(image, svm_boxes, opencv_boxes, annotated_boxes):
    for (x, y, score, scale) in svm_boxes:
        cv2.rectangle(image, (int(x), int(y)), (int(x + 64 * scale), int(y + 128 * scale)), (0, 255, 0), 2)
    for (x, y, w, h) in opencv_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for (x1, y1, x2, y2) in annotated_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imshow("Combined Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Carica il classificatore SVM personalizzato
classifier = load("svm_model.joblib")
if classifier is None:
    print("Errore: modello non caricato")


current_dir = os.path.dirname(os.path.abspath(__file__))

image_directory = os.path.join(current_dir, "Images")

image_path=select_random_image(image_directory)

# Construct the annotation path dynamically based on the test_image_path
base_name = os.path.splitext(os.path.basename(image_path))[0]  # Extracts 'test_image4' from 'test_image4.jpg'
annotation_folder = os.path.join(current_dir, "WiderPerson/Annotations")
annotation_path = os.path.join(annotation_folder, f"{base_name}.jpg.txt")  # Builds the path 'Annotations/test_image4.txt'

# Carica immagine e annotazioni
test_image = cv2.imread(image_path)
annotated_boxes = read_annotations(annotation_path)

# Applica detection SVM e OpenCV
svm_detections = detect_pedestrians_svm(classifier.best_estimator_, test_image, [2, 1, 0.75])
opencv_boxes = detect_pedestrians_opencv(test_image)

# Disegna tutte le detections
draw_detections_combined(test_image, svm_detections, opencv_boxes, annotated_boxes)