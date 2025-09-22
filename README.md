# Titanic_survival_prediction
This project predicts passenger survival on the Titanic using machine learning techniques. By analyzing features such as gender, age, class, fare, and family details, multiple models are trained and compared to identify the best-performing approach. Feature engineering and data preprocessing are applied to improve prediction accuracy.

---

## Dataset

- **train.csv**: Contains passenger details along with survival labels (target variable).
- **test.csv**: Contains passenger details without survival labels; used for generating predictions to submit.

---

## Features Used

- **Demographics:** Age, Sex
- **Ticket Information:** Pclass, Fare, Embarked
- **Family Details:** SibSp, Parch, FamilySize, Family
- **Engineered Features:**  
  - `FamilySize` = SibSp + Parch + 1  
  - `Family` = 0 if FamilySize == 1 (alone), else 1  
  - Binned categories for FamilySize (Small, Medium, Large)

---

## Objectives

- Predict survival status of passengers using machine learning models.
- Identify key features influencing survival.
- Compare multiple models to determine the best approach.
- Generate a submission file for Kaggle.

---

## Methodology

### Data Preprocessing & Feature Engineering
- Handle missing values and outliers.
- Encode categorical features (`Sex`, `Embarked`) using one-hot encoding.
- Scale numerical features as needed.
- Create new features like `FamilySize` and `Family`.

### Models Used
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)

### Workflow
- Use pipelines for consistent preprocessing and model training.
- Hyperparameter tuning with GridSearchCV.
- Model evaluation via cross-validation and metrics such as Accuracy, Precision, Recall, F1-score.
- Save the best-performing model for predictions.

---
### Setup
1. Clone the repository and install dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn

---

## Author
Samar Jahan Burney 
