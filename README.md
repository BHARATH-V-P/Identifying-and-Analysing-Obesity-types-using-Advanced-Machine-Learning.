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

### Baseline Validation
- We assessed models using various preprocessing techniques, including XGB, LGB, and RandomForestClassifier. Preprocessing_1 yielded the highest accuracy for XGB and LGB models, with notable accuracy scores of around 0.9126 for XGB and 0.9127 for LGB.

### Optimized Models
- Bayesian Optimization and Optuna were used to fine-tune the models. The optimized XGB models achieved accuracy scores up to 0.9162, improving log_loss to approximately 0.2438. For LGB models, the best accuracy reached 0.9153, with a log_loss of 0.2465.

### Resampled Data Validation
- Models trained on resampled data showed varied performance. The LGB model with preprocessing_1_resampled achieved an accuracy of 0.9111 and a Cohen score of 0.8958. XGB models had slightly lower accuracy compared to baseline but still performed well.

### Clustered Data Evaluation
- Evaluation on clustered data showed that XGB and LGB models performed best in cluster_2, with accuracies of 0.9719 (XGB) and 0.9704 (LGB). This indicates that the models effectively capture the nuances in clustered data.

Overall, the project demonstrated significant improvements in model performance through optimization techniques and clustering, achieving better accuracy and reduced log_loss.

### Best Performing Model
LightGBM model-: the metrics are: Accuracy Score: 0.92, Cohen's Kappa Score: 0.9851, Log Loss: 0.2445.


## License

   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

   
