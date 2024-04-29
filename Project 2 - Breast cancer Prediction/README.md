## Breast Cancer Prediction Document

### Introduction:
This project focuses on utilizing machine learning to improve breast cancer diagnosis. By employing a Support Vector Classifier (SVC) and leveraging the Breast Cancer Wisconsin (Diagnostic) dataset, we aim to accurately classify tumors as malignant or benign based on diagnostic features. Through rigorous data preprocessing, feature selection, and hyperparameter tuning, we optimize the SVC model's performance. Evaluation metrics such as accuracy, precision, recall, and F1-score provide insights into the model's effectiveness. This work contributes to advancing the application of machine learning in healthcare for more precise and timely diagnosis of breast cancer.

### Data Preprocessing:
1. **Loading Dataset**: The Breast Cancer Wisconsin (Diagnostic) dataset is loaded using `load_breast_cancer()` function from `sklearn.datasets`.
2. **Data Splitting**: The dataset is split into features (X) and target (y). Features are stored in a Pandas DataFrame.
3. **Exploratory Data Analysis (EDA)**:
   - Descriptive statistics and information about features are obtained using `describe()` and `info()` methods respectively.
   - Null values are checked using `isnull().sum()` method.
   - Histograms of target variable 'Diagnosis' and features are plotted using Matplotlib and Seaborn.

### Machine Learning Model Implementation:
1. **Data Scaling**: Standard scaling is performed on features using `StandardScaler` from `sklearn.preprocessing`.
2. **Model Selection**: Support Vector Classifier (SVC) from `sklearn.svm` is chosen as the machine learning model.
3. **Hyperparameter Tuning**: Grid search with cross-validation (5 folds) is employed to find the best hyperparameters for the SVC model (`C`, `gamma`, `kernel`). The grid search is performed using `GridSearchCV` from `sklearn.model_selection`.
4. **Model Training**: The SVC model with the best hyperparameters is trained on the scaled training data.
5. **Model Evaluation**: The trained model is evaluated on the scaled test data using accuracy score, confusion matrix, and classification report. These metrics are obtained using functions from `sklearn.metrics`.

1. **Model Selection:**
- Support Vector Machine (SVM) is chosen for classification due to its effectiveness in high-dimensional data.

**2. Model Training:**
- The data is split into training and testing sets using train_test_split from sklearn.model_selection with a 80/20 split for training and testing respectively.
- StandardScaler is used to normalize feature values in the training and testing sets.
- GridSearchCV is employed to find the optimal hyperparameters for the SVM model. The grid search explores various combinations of kernel functions (linear, rbf, poly) along with cost (C) and gamma parameters.
- The best model is selected based on the highest accuracy score on the training data using 5-fold cross-validation.

**3. Model Evaluation:**
- The chosen SVM model is fit on the training data and used to predict diagnoses on the testing set.
- Model performance is evaluated using the following metrics:
    - Accuracy: Proportion of correctly predicted diagnoses.
    - Confusion Matrix: Visualization of true positives, true negatives, false positives, and false negatives.
    - Classification Report: Detailed precision, recall, F1-score for each class (malignant, benign).

### Model Performance Metrics:
- **Accuracy Score**: It measures the proportion of correctly classified samples. An accuracy score of 0.982 indicates that the model correctly predicts the class of approximately 98.2% of the samples in the test set.
- **Confusion Matrix**: It provides a summary of correct and incorrect predictions made by the model. In this case, there are 41 true negatives, 71 true positives, 2 false negatives, and 0 false positives.
- **Classification Report**: It includes precision, recall, and F1-score for each class (0 and 1). Precision measures the proportion of true positive predictions out of all positive predictions. Recall measures the proportion of true positives that were correctly predicted. F1-score is the harmonic mean of precision and recall. The report indicates high precision, recall, and F1-score for both classes, indicating good performance of the model.

### Challenges Faced:
- Determining the optimal values for hyperparameters (C, gamma, kernel) of the Support Vector Classifier (SVC) can be challenging. Grid search with cross-validation helps in finding the best hyperparameters, but it requires computational resources and time, especially for large datasets or when the search space is extensive.
