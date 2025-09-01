# Titanic Survival Prediction - Logistic Regression Model

A machine learning project that predicts passenger survival on the RMS Titanic using logistic regression. This analysis explores the factors that determined survival during one of history's most famous maritime disasters and demonstrates binary classification techniques.

## Project Overview

This project analyzes the famous Titanic dataset to build a predictive model for passenger survival. The model uses passenger demographics, ticket class, family relationships, and embarkation details to predict survival outcomes, providing insights into the social and economic factors that influenced survival rates during the disaster.

## Objectives

- Predict passenger survival probability based on demographics and ticket information
- Identify the key factors that influenced survival on the Titanic
- Evaluate binary classification model performance
- Provide historical insights into disaster survival patterns

## Dataset Description

The Titanic dataset contains 891 passenger records with the following features:

### Features Used for Prediction

**Passenger Demographics:**
- **Age**: Age of the passenger (filled using class-based imputation)
- **Sex**: Gender of the passenger (encoded: male=1, female=0)

**Ticket & Class Information:**
- **Pclass**: Ticket class (1=First, 2=Second, 3=Third class)
- **Fare**: Passenger fare paid for the ticket

**Family Relationships:**
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard

**Embarkation Details:**
- **Embarked**: Port of embarkation (encoded as Q, S dummy variables)
  - C = Cherbourg, Q = Queenstown, S = Southampton

### Target Variable
- **Survived**: Survival status (0 = Did not survive, 1 = Survived)

### Removed Features
- **PassengerId**: Unique identifier (not predictive)
- **Name**: Passenger name (too specific for modeling)
- **Ticket**: Ticket number (not standardized)
- **Cabin**: Cabin number (too many missing values - 77% missing)

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install dependencies using:
```python
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Structure

```
├── titanic_train.csv          # Training dataset
├── titanic_survival_analysis.py # Main model notebook
└── README.md                  # This file
```

## Usage

1. **Load and explore the data**:
   ```python
   import pandas as pd
   train = pd.read_csv('titanic_train.csv')
   train.head()
   ```

2. **Data preprocessing**:
   ```python
   # Remove unnecessary columns
   train.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
   
   # Handle categorical variables
   sex = pd.get_dummies(train['Sex'], drop_first=True)
   embark = pd.get_dummies(train['Embarked'], drop_first=True)
   
   # Age imputation based on passenger class
   # Class 1: 37 years, Class 2: 29 years, Class 3: 24 years
   ```

3. **Run the complete analysis**:
   ```python
   # The script includes:
   # - Data cleaning and preprocessing
   # - Missing value handling with intelligent imputation
   # - Train/test split (70/30)
   # - Logistic regression training
   # - Model evaluation with multiple metrics
   ```

4. **Make survival predictions**:
   ```python
   # Example prediction for a new passenger
   new_passenger = pd.DataFrame([{
       'Pclass': 3, 'Age': 25, 'SibSp': 0, 'Parch': 0, 
       'Fare': 7.25, 'male': 1, 'Q': 0, 'S': 0
   }])
   prediction = model.predict(new_passenger)
   ```

## Model Performance

The logistic regression model shows solid classification performance:

- **Overall Accuracy**: 76.1%
- **Precision (Did not survive)**: 77%
- **Precision (Survived)**: 73%
- **Recall (Did not survive)**: 86%
- **Recall (Survived)**: 61%
- **F1-Score**: 0.76 (weighted average)

### Confusion Matrix Analysis
```
                Predicted
Actual    Not Survived  Survived
Not Survived    141       23
Survived         41       63
```

**Model Strengths:**
- Good at identifying passengers who did not survive (86% recall)
- Balanced precision between both classes
- Reasonable overall accuracy for historical disaster data

## Historical Insights & Survival Factors

### The "Women and Children First" Protocol
The analysis reveals clear evidence of the maritime evacuation protocol:
- **Gender was crucial**: Survival rates differed dramatically between men and women
- **Age imputation strategy**: Used class-based age estimates (1st class: 37 years, 2nd: 29 years, 3rd: 24 years)

### Class and Social Hierarchy
- **Ticket Class (Pclass)**: Higher class passengers had better survival chances
- **Fare**: Higher fares generally correlated with better survival odds
- **Cabin Location**: Removed due to missing data, but cabin deck likely affected evacuation access

### Family Dynamics
- **SibSp/Parch**: Family size may have influenced survival decisions
- **Traveling alone vs. with family**: Different survival strategies and challenges

## Data Quality & Preprocessing

**Missing Data Handling:**
- **Age**: 177 missing values (19.9%) - filled using class-based imputation
- **Cabin**: 687 missing values (77.1%) - column removed due to excessive missingness
- **Embarked**: 2 missing values - handled during encoding

**Intelligent Age Imputation:**
Based on passenger class analysis:
- **1st Class**: Mean age 37 years
- **2nd Class**: Mean age 29 years  
- **3rd Class**: Mean age 24 years

**Categorical Encoding:**
- **Sex**: One-hot encoded (male=1, female=0)
- **Embarked**: One-hot encoded with drop_first=True (Q, S dummy variables)

## Visualizations

The analysis includes key visualizations:

1. **Missing Data Heatmap**: Visual identification of data quality issues
2. **Survival by Gender**: Count plot showing gender-based survival patterns
3. **Age Distribution**: Histogram of passenger ages after imputation
4. **Age by Class**: Box plot showing age distribution across ticket classes

## Historical Context

**The Titanic Disaster (April 15, 1912):**
- **Total Passengers**: 891 in training dataset
- **Overall Survival Rate**: 38.4% (342 survived out of 891)
- **Maritime Protocol**: "Women and children first" evacuation policy
- **Class System**: First-class passengers had preferential access to lifeboats

## Statistical Summary

| Feature | Mean | Description | Survival Impact |
|---------|------|-------------|-----------------|
| **Survival Rate** | 38.4% | Overall survival rate | Target variable |
| **Age** | 29.7 years | Average passenger age | Age-related factors |
| **Pclass** | 2.31 | Average ticket class | Higher class = better survival |
| **Fare** | $32.20 | Average ticket fare | Higher fare = better survival |
| **Family Size** | 0.90 | Average family members aboard | Mixed impact |

## Model Applications

**Historical Analysis:**
- Understanding social factors in disaster survival
- Maritime safety and evacuation protocol analysis
- Socioeconomic impact studies

**Educational Purposes:**
- Binary classification demonstration
- Feature engineering and data preprocessing examples
- Historical data science case study

**Modern Applications:**
- Emergency evacuation planning
- Risk assessment modeling
- Demographic-based survival analysis

## Key Findings

### Primary Survival Factors
1. **Gender**: Strong predictor based on "women and children first" protocol
2. **Passenger Class**: Social hierarchy significantly affected survival chances
3. **Age**: Younger passengers generally had better survival rates
4. **Fare**: Economic status influenced access to safer areas of the ship

### Data Science Insights
- **Missing Data Strategy**: Class-based age imputation proved effective
- **Feature Selection**: Cabin data too sparse to be useful
- **Categorical Encoding**: Proper dummy variable creation essential
- **Binary Classification**: Logistic regression suitable for survival prediction

## ⚠ Model Limitations

- **Historical Context**: Based on unique disaster circumstances
- **Limited Generalizability**: Specific to Titanic disaster conditions
- **Missing Variables**: Doesn't include cabin location, crew instructions, or evacuation timing
- **Moderate Accuracy**: 76% accuracy suggests additional factors influenced survival
- **Class Imbalance**: More non-survivors than survivors in dataset

## Future Enhancements

**Model Improvements:**
- **Feature Engineering**: Create family size, title extraction from names
- **Advanced Algorithms**: Try Random Forest, XGBoost, or Neural Networks
- **Cross-Validation**: Implement stratified k-fold validation
- **Ensemble Methods**: Combine multiple algorithms for better performance

**Additional Analysis:**
- **Survival Rate by Deck**: If cabin data becomes available
- **Family Survival Patterns**: Analyze survival within family groups
- **Embarkation Analysis**: Deeper dive into port-specific patterns
- **Time-Series Analysis**: If evacuation timing data available

## Contributing

Contributions welcome! Areas for improvement:
- Advanced feature engineering (title extraction, family grouping)
- Alternative classification algorithms
- Interactive survival probability calculator
- Historical disaster comparison analysis

## Educational Value

**Learning Objectives:**
- Binary classification with logistic regression
- Handling missing data with domain knowledge
- Categorical variable encoding techniques
- Model evaluation for classification problems
- Historical data analysis and interpretation

**Key Concepts Demonstrated:**
- Data preprocessing pipeline
- Intelligent missing value imputation
- Feature engineering for categorical variables
- Classification metrics interpretation
- Historical insights from data analysis

## Technical Notes

**Preprocessing Pipeline:**
1. Remove non-predictive columns (PassengerId, Name, Ticket)
2. Handle missing Cabin data (drop due to 77% missingness)
3. Intelligent age imputation based on passenger class
4. One-hot encode categorical variables (Sex, Embarked)
5. Train/test split: 70/30 with random_state=43

**Model Configuration:**
- Algorithm: Logistic Regression (binary classification)
- No regularization parameters specified
- All processed features included in final model

**Example Prediction:**
```
New Passenger Profile:
- 3rd Class, 25-year-old male
- Traveling alone, fare $7.25
- Embarked from Cherbourg
Prediction: Did not survive
```

---

**Model Accuracy**: 76.1% | **Historical Dataset**: RMS Titanic (1912) | **Survival Rate**: 38.4%
