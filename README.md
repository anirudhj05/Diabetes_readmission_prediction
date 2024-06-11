# Optimizing Diabetes Management: Analyzing Medication Efficacy and Prescription Practices for Reduced Readmission Rates

## Project Overview

This project focuses on the prediction and analysis of hospital readmission rates for patients with Type 2 Diabetes Mellitus (T2DM). Leveraging a robust dataset and various ML techniques, we aim to identify key factors influencing readmission rates. The goal is to provide actionable insights that can help tailor patient management strategies to reduce readmissions and improve overall treatment outcomes.

### Objectives
- Develop predictive models that integrate patient demographics, clinical data, and treatment histories to predict hospital readmissions.
- Compare the efficacy of different ML algorithms in handling complex, heterogeneous medical data.
- Extract insights from the data to inform better medication practices and intervention strategies.

## Dataset Description

The dataset includes comprehensive information from 130 U.S. hospitals, collected between 1999 and 2008, featuring 101,766 records of patients diagnosed with T2DM. This dataset reflects a wide range of variables:

### Key Features
- **Demographic Information**: Age, gender, race.
- **Clinical Data**: Number of inpatient visits, lab test results, and previous diagnoses.
- **Treatment Data**: Details on the type of medication prescribed, changes in medication, and whether the medication was administered before the hospital stay.

### Data Challenges and Solutions
- **Missing Data**: Significant fields like 'weight' and 'payer code' had over 50% missing data. For 'weight', due to the high percentage of missing data, the field was removed. For 'payer code', missing values were imputed using the mode of the column.
- **High Cardinality**: Fields like medical specialty had 84 distinct values which were reduced using dimensionality reduction techniques to consolidate similar specialties.
- **Class Imbalance**: The dataset exhibited a skewed distribution in the target variable (readmission). We applied the Synthetic Minority Oversampling Technique (SMOTE) to balance it, improving the model's performance and its ability to generalize.

## Methodology

### Preprocessing Steps
- **Data Cleaning**: Removed columns with extensive missing data and applied imputation techniques for other missing values.
- **Feature Encoding**: Encoded categorical variables using one-hot encoding for nominal data and ordinal encoding for ordinal data, such as age groups.
- **Data Normalization**: Scaled numerical features to have a zero mean and a unit variance to aid in the convergence of the neural networks.

### Model Development
We employed several machine learning models to address the prediction task:

1. **Feed Forward Neural Network (FFNN)**: Designed to capture non-linear interactions between features through multiple layers of neurons.
2. **Residual Neural Network (ResNet)**: Utilizes skip connections to handle deeper network architectures without suffering from the vanishing gradient problem.
3. **Logistic Regression (LR)** and **XGBoost**: Used as baseline models for comparison purposes.

#### Hyperparameters
- **FFNN Configuration**:
  - **Layers**: Input layer, 2 hidden layers with 128 and 64 neurons, respectively, and an output layer.
  - **Activation**: ReLU for hidden layers and sigmoid for the output layer.
  - **Optimizer**: Adam with a learning rate of 0.0005.
  - **Regularization**: Dropout of 0.2 and L2 regularization with a coefficient of 0.01.
  - **Training**: 200 epochs with early stopping based on validation loss.

- **ResNet Configuration**:
  - **Layers**: Series of residual blocks with transition layers.
  - **Activation**: ReLU throughout the network.
  - **Optimizer**: Adam with an initial learning rate of 0.001, adjusted by ReduceLROnPlateau on plateau.
  - **Regularization**: Increasing dropout from 0.3 to 0.5 through the blocks.
  - **Training**: 200 epochs with early stopping.

### Evaluation Metrics
- Models were evaluated based on accuracy, precision, recall, and F1-score. The performance metrics helped us gauge the efficacy of each model in predicting the readmission accurately.

## Results

- **FFNN Performance**: Achieved an accuracy of 75.88%, highlighting its ability to model complex patterns effectively.
- **LR Performance**: Provided a baseline accuracy of 75.70%, demonstrating the importance of linear models in certain scenarios.
- **ResNet Performance**: Showed a lower accuracy of 71.37%, suggesting challenges in adapting deep network architectures to structured medical data.

### Visualizations and Analysis
- **Cumulative Gain and Lift Charts**: Illustrated how each model prioritized the correct identification of readmitted patients, with FFNN showing the most promising results.
- **Insights from Model Outputs**: Detailed analysis of feature importance and model decisions provided insights into the critical factors influencing readmissions, aiding in the refinement of treatment strategies.

