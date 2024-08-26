# Identifying-and-Analysing-Obesity-types-using-Advanced-Machine-Learning.

This project aims to build a classification model to predict an individual’s health category based on several independent variables such as height, weight, frequency of water consumption, and calorie intake. While weight and height are primary indicators of overall health, the dataset and model explore how additional factors, including dietary habits and physical activity, influence general health. The project involves creating insightful visualizations and optimizing machine learning models to provide a comprehensive analysis of these factors and their impact on health status.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [License](#license)

## Introduction

This project aims to build a classification model to accurately predict the health category an individual belongs to, such as underweight, normal weight, obesity, and its subclasses. The goal is to optimize the model and handle any overfitting issues using techniques like Bayesian optimization, Grid Search CV, and Optuna.
The dataset is diverse, with only minor imbalances. The project examines significant variability in the dependent variable induced by numerical variables using ANOVA and the Shapiro-Wilk test. The primary machine learning models used are Random Forest Classifier, XGB Classifier, and LGB Classifier. Further improvements to the models were aimed to be achieved through K Prototypes clustering.

## Installation

1. Clone The repository:
   git clone https://github.com/BHARATH11112222/Identifying-and-Analysing-Obesity-types-using-Advanced-Machine-Learning.

2. Navigate to project directory:
   cd Identifying-and-Analysing-Obesity-types-using-Advanced-Machine-Learning.

3. Install dependencies:
   pip install -r requirements.txt

## Usage

1. Prepare Your Dataset:
   - Place your datasets in the data/ directory.
   -Ensure the dataset is named obesity_train.csv, Obesity_original_Dataset.csv, or adjust the script accordingly.

2. Open the Jupyter Notebook:
   - Launch the notebook: Identifying_and_Analysing_Obesity_types_using_Advanced_Machine_Learning.ipynb.

3. Notebook Workflow:
   - Data Preprocessing:
      - Check for and handle null values, duplicate records, and other discrepancies within the raw data.
   - BMI Calculation & Classification:
      - Calculate BMI for individuals, classify them into health categories, and compare these classifications with actual health categories from the dataset to          evaluate accuracy.
   - Feature Impact Analysis:
      - Analyze the impact of other features on obesity using machine learning models, providing insights and recommendations based on findings.
   - Exploratory Data Analysis (EDA):
      - Perform EDA to examine features that affect specific obesity classes, observing significant patterns or changes in feature distributions.
   - Feature Engineering & Selection:
      - Apply necessary feature engineering and perform feature selection using RFECV (Recursive Feature Elimination with Cross-Validation).
   - KMeans Prototyping:
      - Identify the ideal number of clusters for KMeans and assess how different models learn from these clusters.
   - Hyperparameter Tuning:
      - Use Bayesian Optimization, Optuna, and GridSearchCV to optimize model performance.
   - Feature Importance:
      - Prioritize feature importance for individual models to understand their impact.
   - KPI Metrics:
      - Track specific key performance indicators (KPIs), such as R2 score, to evaluate model behavior on different preprocessed datasets.
        
4. Review the Results:

   - Evaluation Metrics: Found in results/evaluation_metrics.txt.
   - Predictions: Found in results/predictions.csv.


## Features
   - Data Preprocessing: Handles missing values and normalization to prepare the data for modeling.
   - Insightful Visualizations: Provides visualizations to better understand the imbalanced dataset.
   - Class Label Agreement Analysis: Compares assigned class labels in the dataset with labels derived solely from BMI calculations.
   - Statistical Overview: Offers a statistical summary of the features involved in the analysis.
   - Multiple Preprocessing Techniques: Explores three different preprocessing methods to evaluate their impact on model performance.
   - Model Training: Utilizes machine learning algorithms such as Random Forest Classification, XGBoost, and LightGBM.
   - Custom Metrics Creation: Develops custom metrics to enhance the performance of Optuna and Bayesian Optimization.
   - Clustering Analysis: Investigates the clustering process and its effect on model performance through both global and cluster-specific analyses.
   - Overfitting Reduction: Implements strategies to reduce overfitting, improving the model's generalization to different classes.
   - Model Evaluation: Evaluates models based on the quality of fitting during data training, using various evaluation metrics.

## Results

## Understanding the Agreement Between Assigned Class Labels and BMI-Based Class Labels

### 1. Classification Report

- The classification metrics show varying levels of performance across different classes, with high precision and recall for Insufficient_Weight and Obesity_Type_III.
- The overall accuracy (0.37) and the macro and weighted averages suggest that the BMI-based labels do not align well with the dataset target labels.

### 2. Scoring Metrics

- **Accuracy Score**: 0.373157
- **Cohen's Kappa Score**: 0.872570

These scores indicate a moderate agreement between the BMI-based labels and the assigned class labels.

### 3. Interpretation

- The BMI-based labels do not perform well across many classes, with particularly low precision and recall for several obesity types. This suggests that BMI alone may not be a strong predictor for these categories, highlighting the need for more comprehensive features or models.

## Correlation and ANOVA Analysis

### 1. Shapiro-Wilk Test Results

- All categories' numerical features are not normally distributed, which is significant for selecting appropriate statistical tests and models.

### 2. Interpretation

- Since the numerical features do not follow a normal distribution, non-parametric methods or transformations may be required for further analysis.

## Baseline Model Validation

### 1. Performance Metrics

- **Top Performing Models**:
  - Preprocessing 1: XGB and LGB
  - Preprocessing 2: LGB
  - Preprocessing 3: XGB

### 2. Interpretation

- The XGB and LGB models performed similarly well across different preprocessing techniques, indicating their robustness. The choice of preprocessing method can impact performance but not drastically.

## Model Training Using Optimized Hyperparameters

### 1. Performance Metrics

- **XGB Classifier**:
  - Highest accuracy and Cohen's Kappa scores were achieved with preprocessing 2 and 3 configurations.
- **LGB Classifier**:
  - Best performance was noted with preprocessing 2 using Optuna.

### 2. Interpretation

- Optimizing hyperparameters significantly improved model performance, with XGB and LGB showing strong results. The choice of configuration can greatly impact the model’s accuracy and agreement.

## Performance Metrics by Resampling Method

### 1. Metrics for Random Forest, XGB, and LGB

- **XGB Classifier**:
  - RandomUnderSampler and RandomOverSampler showed better accuracy compared to other methods.
- **LGB Classifier**:
  - ADASYN and RandomOverSampler were the most effective methods.

### 2. Interpretation

- The effectiveness of resampling methods varies across models. For XGB, RandomUnderSampler performed best, while for LGB, ADASYN was more effective.

## Validating Resampled Datasets

### 1. Performance Metrics

- **Top Performing Models**:
  - Preprocessing 1 Resampled: XGB
  - Preprocessing 2 Resampled: LGB
  - Preprocessing 3 Resampled: LGB

### 2. Interpretation

- Resampling did not lead to a significant improvement in performance. It’s crucial to assess whether resampling adds value or introduces more noise.

## Clustering Inference and Recommendations

### 1. Preprocessing 1, 2, and 3

- Different numbers of clusters (6 for preprocessing 1, 7 for preprocessing 2, and 6 for preprocessing 3) were recommended based on WCSS and silhouette scores.

### 2. Interpretation

- A balanced approach considering WCSS and silhouette scores is essential for effective clustering. The recommended cluster numbers should help in capturing underlying patterns without overfitting.

## PCA Component Weights and Percentage Contribution

### 1. Key Features

- Features like Height, Weight, and Age contribute significantly to the principal components, indicating their importance in the clustering and overall model performance.

### 2. Interpretation

- The high contribution of features like Height and Weight suggests they play a crucial role in the clustering and classification tasks. Feature importance should guide further feature engineering and model refinement.

This detailed analysis offers insights into how well the models and clustering methods are performing, along with recommendations for improving feature engineering and model performance.

## License

   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

   
