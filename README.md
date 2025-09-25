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
Prepare your dataset (train.csv and test.csv) in the project directory.
Training
Run the training script:

CopyRun
python train.py
Prediction & Submission
Generate predictions on the test set and save submission:

CopyRun
# Example code snippet
import pandas as pd
# Load test data
test = pd.read_csv('test.csv')
# Load trained model
# model = ... (load your best model)
# Generate predictions
# test_pred = model.predict(test_processed)
# Save submission
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_pred
})
submission.to_csv("submission.csv", index=False)
Results & Model Performance
Model	Accuracy	F1-score	Notes
Logistic Regression	0.81	Good baseline, interpretable
Decision Tree	0.84	Simple, prone to overfitting
Random Forest	0.84	Robust, ensemble method
Gradient Boosting 0.85	Best performer, captures complex interactions
XGBoost	0.84	Similar to Gradient Boosting
LightGBM	0.82	Efficient training
SVC	0.81	Underperformed, sensitive to hyperparameters
KNN	0.80	Underperformed compared to ensembles
Visualizations
Confusion matrices
Feature importance plots
Distribution of features and survival rates
Conclusion
Ensemble models like Gradient Boosting and XGBoost yield the best results for Titanic survival prediction. Feature engineering, especially FamilySize and Family, along with proper handling of missing data, enhances model performance. Logistic Regression provides a strong baseline with interpretability.

License
This project is for educational purposes. Feel free to adapt and improve upon it.

Acknowledgments
Titanic dataset from Kaggle
scikit-learn, XGBoost, LightGBM documentation
Data visualization with Matplotlib and Seaborn

---

## Author
Samar Jahan Burney 
