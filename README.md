# Customer-Churn-Prediction
Customer Churn Prediction
Customer Churn Prediction
This repository contains a machine learning project to predict customer churn for a bank using structured customer data such as demographics, account information, and product usage.[1]

Project Overview
Customer churn is a critical business problem where customers leave a service, resulting in revenue loss and higher acquisition costs.[1] This project builds and evaluates classification models to identify customers with a high likelihood of exiting, enabling proactive retention strategies.[1]

Dataset
Source: Bank churn modelling dataset (10,000 rows × 14 columns).[1]
Key features:
Demographics: Geography, Gender, Age, Tenure.[1]
Account info: CreditScore, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.[1]
Target variable: Exited (1 = customer churned, 0 = retained).[1]
Note: The notebook currently loads the CSV from a local path; update this path or place the dataset in the project folder when running the code.[1]

Approach
Data loading and exploration
Load the CSV into a pandas DataFrame and inspect structure, distributions, and missing values.[1]
Preprocessing
Encode categorical variables such as Geography and Gender using label encoding.[1]
Separate features and target, then split into training and test sets using train_test_split.[1]
Apply feature scaling using StandardScaler for numerical features to improve model performance.[1]
Modeling
Train multiple classification models, including:
Logistic Regression.[1]
Random Forest Classifier.[1]
Gradient Boosting Classifier.[1]
Evaluation
Evaluate models on the test set using:
Accuracy score.[1]
Classification report (precision, recall, f1-score).[1]
Confusion matrix to analyze correct and incorrect predictions.[1]
Repository Structure
Churn-Prediction.ipynb – Main Jupyter Notebook with data preprocessing, model training, and evaluation.[1]
data/Churn_Modelling.csv – Dataset file (recommended relative path; add this folder and file when publishing).[1]
Requirements
Key Python libraries used:

pandas, numpy.[1]
scikit-learn (model_selection, preprocessing, metrics, linear_model, ensemble).[1]
matplotlib (for basic plots, if used in EDA).[1]
Install them with:

pip install -r requirements.txt
How to Run
Clone the repository:
git clone
cd
Set up environment and install dependencies:
python -m venv venv
source venv/bin/activate (Linux/Mac)
venv\Scripts\activate (Windows)
pip install -r requirements.txt
Place the dataset:
Create a data/ folder in the project root and add Churn_Modelling.csv.[1]
Update the notebook path to: df = pd.read_csv("data/Churn_Modelling.csv").[1]
Run the notebook:
jupyter notebook
Open Churn-Prediction.ipynb and run all cells in order.
Future Improvements
Add automated hyperparameter tuning (GridSearchCV or RandomizedSearchCV).[1]
Perform more detailed EDA and feature engineering (e.g., interaction features, binning of continuous variables).[1]
Log experiments and metrics using MLflow or similar tools.
License
This project respects all relevant intellectual property and copyright for datasets and libraries used.
Specify your chosen open-source license here (e.g., MIT License) depending on your needs.

1

About
Customer Churn Prediction

Resources
 Readme
 Activity
Stars
 0 stars
Watchers
 0 watching
Forks
 0 forks
Releases
No releases published
Create a new release
Packages
No packages published
Publish your first package
Languages
Jupyter Notebook
100.0%
Footer
© 2025 GitHub, Inc.
Footer navigation
Te
