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
1. Prepare your dataset: Place your datasets in the data/ directory. Ensure it is named OBESITY TRAIN.csv, ObesityDataSet.csv or adjust the script accordingly.
3. Open the Jupyter Notebook: jupyter notebook "OBESITY_CONCISE_EDA.ipynb"
   The script will:
   - Preprocess the raw data by checking for null values, duplicate values and any other discrepencies within the raw data.
   - Calculates BMI for individuals in a dataset, classifying them into health categories based on this metric, and comparing these classifications 
     to actual health categories based on the available data to evaluate accuracye between the two.
   - It analyzes the impact of other features on obesity using machine learning models and provides insights and recommendations based on the findings.
   - Perform relevant EDA to get a better look on features that effect specific classes individually by observing for significant patterns or changes in the 
     distribution of features corresponding to each individual class.
   - Perform neccessary feature engineering and feature seleciton using RFECV.
   - Performs KMeans Clustering by identifyin the ideal no. of clusters and how different model learns from different clusters.
   - Perform Hyperparameter tuning through Bayesian Optimization, Optuna and GridSearchCV as needed to try and improve model performance
   - Prioritize feature  importance for individual models.w
   - Keep track of specific KPI metrics such as R2 score to understand model behavior for specific preprocessed datasets
5. Review the results:
   Evaluation Metrics: Found in results/evaluation_metrics.txt
   Predictions: Found in results/predictions.csv

## Features
   - Data Preprocessing: Handles missing values and  normalization.
   - Insightful Visualizations to better understand the imabalanced dataset.
   - Understanding the agreement between assigned class labels in the dataset and class labels solely based on just BMI
   - Get a statistical overview of the features involved
   - Explores 3 different preprocessing techniques to evaluate how they effect model performance
   - Model Training: Utilizes Random Forest Classification, XGBoost and LGB for machine learning.
   - Create custom metrics to aid better results from Optuna and Bayesian Optimization
   - Understanding the process of clustering and its effect on model peformance by conducting both global specific and cluster specific analysis
   - Attempts to reduce overfitting of the model to specific classes thus improving generalization
   - Model Evaluation based on the quality of fitting during data trianing

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

   
