import os
from joblib import load
import numpy as np
from estrazionedati import datiPositivi, datiNegativi
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def read_values_from_txt(txt_file):
    with open(txt_file, 'r') as file:
        values = [line.strip() for line in file]
    return values

current_dir = os.path.dirname(os.path.abspath(__file__))

# Impostazioni
annotation_folder = os.path.join(current_dir, "Annotations")
images_folder = os.path.join(current_dir, "Images")
desired_size = (64, 128)

val_assignment_file = os.path.join(current_dir, "split_positive/val_assignment.txt")
negative_images_folder_val = os.path.join(current_dir, "negative/val_neg/val_neg")
val_values = read_values_from_txt(val_assignment_file)

# Carica il modello addestrato
lista_modelli = []
#prendi i modelli
model_path = os.path.join(current_dir, "svm_model.joblib")
classifier = load(model_path)



# Estrai le caratteristiche dalle immagini di validazione
positive_features_val, positive_labels_val, annotated_boxes = datiPositivi(annotation_folder, images_folder, desired_size, val_values)
negative_features_val = datiNegativi(negative_images_folder_val,0.1)

X_val = np.vstack((positive_features_val, negative_features_val))
y_val = np.hstack((np.ones(len(positive_features_val)), np.zeros(len(negative_features_val))))

# Fai previsioni sulle caratteristiche delle immagini di validazione
predictions_val = classifier.predict(X_val)

# Valuta le prestazioni del modello
accuracy_val = accuracy_score(y_val, predictions_val)
precision_val = precision_score(y_val, predictions_val)
recall_val = recall_score(y_val, predictions_val)
f1_val = f1_score(y_val, predictions_val)
roc_auc_val = roc_auc_score(y_val, predictions_val)

# Stampare le metriche di valutazione
print("Accuracy on validation set:", accuracy_val)
print("Precision on validation set:", precision_val)
print("Recall on validation set:", recall_val)
print("F1-score on validation set:", f1_val)
print("ROC AUC Score on validation set:", roc_auc_val)