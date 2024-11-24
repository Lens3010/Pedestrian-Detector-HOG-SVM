
# Pedestrian Detector Project

## Project Overview

This project focuses on developing a **Pedestrian Detector** using image processing and machine learning techniques. The main goal is to recognize and outline pedestrians in an input image. The project uses Python scripts and machine learning models to implement and validate the system.

---

## Project Structure

- **`estrazionedati.py`**: This script is responsible for data preparation. It processes images by:
  - Cropping and resizing positive samples.
  - Randomly sampling negative samples.
  - Generating feature descriptors (using HOG).

- **`validazione.py`**: Implements validation of the model using pre-split datasets and metrics like accuracy, precision, recall, and F1-Score.

- **`Validazione_cv.py`**: Implements cross-validation techniques to evaluate model performance more robustly across multiple folds of the dataset.

- **`testing.py`**: Used for testing the trained models on unseen data to analyze their generalization performance.

- **Machine Learning Models**:
  - **`svm_model.joblib`**: The primary trained model using a linear SVM classifier.
  - **`svm_model_centralized.joblib`**: A centralized version of the trained model.
  - **`svm_model.joblib2`**: A secondary variant for comparative testing.

---

## Key Functionalities

1. **Data Preprocessing**:
   - Images are resized and processed to ensure consistency in dimensions.
   - Feature extraction is performed using HOG (Histogram of Oriented Gradients).

2. **Model Training**:
   - A linear SVM classifier is trained using labeled data (pedestrians and non-pedestrians).
   - Various hyperparameters (e.g., `step`, `scale`, `threshold`) are fine-tuned to achieve optimal detection performance.

3. **Sliding Window**:
   - A sliding window mechanism is implemented to allow the classifier to detect pedestrians across multiple scales.

4. **Non-Maximum Suppression**:
   - Post-processing technique to eliminate redundant bounding boxes and highlight the most probable detections.

5. **Evaluation Metrics**:
   - The models are evaluated using standard metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## Performance Results

| Classifier  | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------------|----------|-----------|--------|----------|---------|
| Model 1     | 91.36%   | 89.68%    | 91.46% | 90.56%   | 91.37%  |
| Model 2     | 87.52%   | 86.55%    | 82.96% | 84.72%   | 86.87%  |
| Model 3     | 90.42%   | 88.32%    | 89.30% | 87.77%   | 89.03%  |

---

## Usage Instructions

1. **Dependencies**:
   - Install required libraries:
     ```bash
     pip install -r requirements.txt
     ```
     *(Note: Ensure OpenCV, scikit-learn, and joblib are installed.)*

2. **Running Scripts**:
   - **Data Extraction**: Run `estrazionedati.py` to process and prepare data.
   - **Validation**: Execute `validazione.py` or `Validazione_cv.py` to validate the model.
   - **Testing**: Use `testing.py` to evaluate model performance on test data.

3. **Model Loading**:
   - Load pre-trained models (e.g., `svm_model.joblib`) using:
     ```python
     from joblib import load
     model = load('svm_model.joblib')
     ```

---

## Project Highlights

- Multi-scale pedestrian detection using HOG and SVM.
- Robust model evaluation using cross-validation and real-world data testing.
- Collaborative development with team members contributing diverse approaches for improved results.

## Authors

- Francesco Lena  
- Giuseppe Giuliani  
- Federico Princiotta Cariddi  

--- 

This README provides an overview of the project's functionality and structure, enabling easy navigation and use of the provided scripts and models.
