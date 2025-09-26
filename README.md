# Titanic_survival_prediction
This project predicts passenger survival on the Titanic using machine learning techniques. By analyzing features such as gender, age, class, fare, multiple models are trained and compared to identify the best-performing approach. Feature engineering and data preprocessing are applied to improve prediction accuracy.

---

## Dataset

- **train data set.csv**: Contains passenger details along with survival labels (target variable).

---

## Objectives

- Predict survival status of passengers using machine learning models.
- Identify key features influencing survival.
- Compare multiple models to determine the best approach.

---

## Methodology

### Data Preprocessing , Feature Engineering & Data cleaning
- Handle missing values and outliers.
- Encode categorical features ('Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title','Deck','FamilySize','IsAlone'`) using one-hot encoding.
- 
- Show barplot and graph before and after cleaning


### Models Used
- Logistic Regression
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
Prepare your dataset (titanic survival data set.csv) in the project directory.
Training
Run the training script:

CopyRun
colab Titanic_train_data.pynb
Prediction & Submission
Generate predictions on the test set and save

Results & Model Performance
Model Accuracy Summary:
                  Model  Accuracy
0  Logistic Regression  0.843575
1        Random Forest  0.793296
2    Gradient Boosting  0.815642
3                  SVC  0.832402
4                  KNN  0.804469
5              XGBoost  0.815642
6             LightGBM  0.798883

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
scikit-learn
Data visualization with Matplotlib and Seaborn

---

## Author
Samar Jahan Burney 
