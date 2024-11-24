import os
import cv2
import joblib
import numpy as np
from testing import detect_pedestrians_svm


def iou(box1, box2):

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area

def read_annotations(annotation_path):
    """Leggi le annotazioni da un file e restituisci una lista di bounding boxes, gestendo il formato con spazi."""
    boxes = []
    with open(annotation_path, 'rb') as file:  # Apri il file in modalità binaria
        content = file.read().decode('utf-8', errors='ignore').replace('\0', '')  # Decodifica ignorando errori e rimuovi caratteri nulli
    for line in content.splitlines():
        if line.strip():  # Verifica che la riga non sia vuota
            parts = line.strip().split()  # Dividi la riga in base agli spazi
            if len(parts) >= 5:  # Assicurati che ci siano almeno 5 parti (ID e coordinate x1, y1, x2, y2)
                try:
                    # Converti solo le parti delle coordinate in interi, ignorando l'ID se presente
                    box = list(map(int, parts[1:5]))
                    boxes.append(box)
                except ValueError as e:
                    print(f"Skipping line with error: {e} - {line}")
    return boxes

def compute_detection_metrics(ground_truth_boxes, predicted_boxes):
    TP = 0
    FP = 0
    FN = 0
    
    matched_gt = []


    # Calcolo dei TP e FP
    for pred_box in predicted_boxes:
        match_found = False
        for gt_box in ground_truth_boxes:
            if iou(pred_box, gt_box) > 0.5:
                if gt_box not in matched_gt:
                    TP += 1
                    matched_gt.append(gt_box)
                    match_found = True
                    break
        if not match_found:
            FP += 1

    # Calcolo dei FN
    FN = len(ground_truth_boxes) - len(matched_gt)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


# Usiamo queste metriche nel loop
current_dir = os.path.dirname(os.path.abspath(__file__))
negative_images_folder_val = os.path.join(current_dir, "negative/val_neg")
annotation_folder = os.path.join(current_dir, "Annotations")
images_folder = os.path.join(current_dir, "Images")
image_path = os.path.join(current_dir, "Images/000040.jpg")
image = cv2.imread(image_path)
desired_size = (64, 128)
val_assignment_file = os.path.join(current_dir, "split_positive/val_assignment.txt")

classifier_path1 = os.path.join(current_dir, "svm_model.joblib")
classifier_path2 = os.path.join(current_dir, "svm_model2.joblib")
classifier1 = joblib.load(classifier_path1)
classifier2 = joblib.load(classifier_path2)

ground_truth_boxes = read_annotations(annotation_folder)
scales = [2, 1, 0.75]

detected1 = detect_pedestrians_svm(classifier1, image, scales)
detected2= detect_pedestrians_svm(classifier2, image, scales)

# Calcola le metriche usando la nuova funzione
precision_val1, recall_val1, f1_val1 = compute_detection_metrics(ground_truth_boxes, detected1)
precision_val2, recall_val2, f1_val2 = compute_detection_metrics(ground_truth_boxes, detected2)

print(f1_val1)
print(f1_val2)