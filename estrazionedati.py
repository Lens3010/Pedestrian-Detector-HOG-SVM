import cv2
import os
import random
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import GridSearchCV
from joblib import dump

def read_values_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        values = [line.strip() for line in file]
    return values



# Percorsi ai file txt
train_assignment_file = 'split_positive/train_assignment.txt'
test_assignment_file = 'split_positive/test_assignment.txt'
val_assignment_file = 'split_positive/val_assignment.txt'

# Leggi i valori dai file txt
train_values = read_values_from_txt(train_assignment_file)
test_values = read_values_from_txt(test_assignment_file)
val_values = read_values_from_txt(val_assignment_file)



def datiPositivi(annotation_folder, images_folder, desired_size, train_values):
    positive_features = []
    positive_labels = []
    annotated_boxes = []  # Lista delle bounding box annotate

    for image_name in os.listdir(images_folder):
        annotation_path = os.path.join(annotation_folder, f"{image_name}.txt")
        image_path = os.path.join(images_folder, image_name)
        
        if image_name[:-4] in train_values:
            image = cv2.imread(image_path)
            if os.path.isfile(annotation_path):
                with open(annotation_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines[1:]:
                        parts = line.split()
                        class_label = int(parts[0])
                        x1, y1, x2, y2 = map(int, parts[1:])
                        if class_label == 1 and x1 < x2 and y1 < y2:
                            w = x2 - x1
                            h = y2 - y1
                            ratio = h / w
                            if 1.0 < ratio < 3.5:
                                pedestrian_region = image[y1:y2, x1:x2]
                                resized_pedestrian_region = cv2.resize(pedestrian_region, desired_size)
                                features = cv2.HOGDescriptor().compute(resized_pedestrian_region).reshape(-1)
                                positive_features.append(features) 
                                positive_labels.append(1)
                                annotated_boxes.append((x1, y1, x2, y2))  # Aggiungi la bounding box annotata

    return positive_features, positive_labels, annotated_boxes

def datiNegativi(negative_images_folder):
    
    negative_features = []

    for file in os.listdir(negative_images_folder):
        if file.endswith((".jpg", ".png")):
            image_path = os.path.join(negative_images_folder, file)
            image = cv2.imread(image_path)
            for _ in range(140):
                height, width, _ = image.shape
                x_start = random.randint(0, width - 64)
                y_start = random.randint(0, height - 128)
                x_end = x_start + 64
                y_end = y_start + 128
                cropped_image = image[y_start:y_end, x_start:x_end]
                features = cv2.HOGDescriptor().compute(cropped_image).reshape(-1)
                negative_features.append(features)  # No flatten here

                # Salvataggio dell'immagine croppata nella cartella corrispondente
                

    return negative_features

def train_svm(X_train, y_train):
    svm = LinearSVC()
    parameters = {'C': [0.1, 1, 10, 100]}
    clf = GridSearchCV(svm, parameters, cv=3)
    clf.fit(X_train, y_train)
    return clf

# Impostazioni
annotation_folder = "Annotations"
images_folder = "Images"
negative_images_folder = "negative/train_neg"
negative_images_folder_test = "negative/test_neg"

desired_size = (64, 128)
train_values = read_values_from_txt(train_assignment_file)

# Estrazione delle caratteristiche
positive_features, positive_labels,annotated_boxes = datiPositivi(annotation_folder, images_folder, desired_size, train_values)
negative_features = datiNegativi(negative_images_folder)

positive_features_test, positive_labels_test,annotated_boxes = datiPositivi(annotation_folder, images_folder, desired_size, test_values)
negative_features_test = datiNegativi(negative_images_folder_test)

X_train = np.vstack((positive_features, negative_features))
y_train = np.hstack((np.ones(len(positive_features)), np.zeros(len(negative_features))))

X_test = np.vstack((positive_features_test, negative_features_test))
y_test = np.hstack((np.ones(len(positive_features_test)), np.zeros(len(negative_features_test))))

# Addestra il classificatore SVM
classifier = train_svm(X_train, y_train)

model_save_path = "svm_model.joblib"

# Salva il modello addestrato
dump(classifier, model_save_path)

print("Modello addestrato salvato con successo!")

# Valuta le prestazioni del modello
accuracy = classifier.score(X_test, y_test)
print("Accuracy:", accuracy)

print("Caratteristiche positive estratte:", len(positive_features))
print("Caratteristiche negative estratte:", len(negative_features))
