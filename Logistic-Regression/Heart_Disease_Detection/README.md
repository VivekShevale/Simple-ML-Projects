# Heart Disease Prediction - Logistic Regression Model

A machine learning project that predicts the presence of heart disease using logistic regression based on key health indicators. This analysis helps identify patients at risk and supports clinical decision-making for cardiovascular health assessment.

## Project Overview

This project analyzes medical data to build a predictive model for heart disease diagnosis. The model uses essential health metrics including age, gender, blood pressure, and cholesterol levels to predict the likelihood of heart disease, providing valuable support for medical screening and early detection programs.

## Objectives

- Predict heart disease presence based on key health indicators
- Identify the most significant risk factors for cardiovascular disease
- Evaluate binary classification performance for medical diagnosis support
- Provide insights for preventive healthcare and risk assessment

## Dataset Description

The heart disease dataset contains 270 patient records with the following features:

### Features Used for Prediction

**Demographic Information:**
- **age**: Patient age in years (29-77 years)
- **sex**: Patient gender (encoded: 0=female, 1=male)

**Cardiovascular Risk Factors:**
- **BP**: Blood pressure (systolic) in mmHg (94-200 range)
- **cholestrol**: Serum cholesterol level in mg/dl (126-564 range)

### Target Variable
- **heart disease**: Presence of heart disease (0=No disease, 1=Heart disease present)

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
├── heart_v2.csv               # Dataset file
├── model_train.ipynb  # Main model notebook
└── README.md                  # This file
```

## 🚀 Usage

1. **Load and explore the data**:
   ```python
   import pandas as pd
   ht = pd.read_csv('heart_v2.csv')
   ht.head()
   ```

2. **Data exploration**:
   ```python
   # Check data quality
   ht.info()
   ht.describe()
   ht.isnull().sum()  # No missing values!
   ```

3. **Run the complete analysis**:
   ```python
   # The script includes:
   # - Exploratory data analysis
   # - Train/test split (75/25)
   # - Logistic regression training
   # - Comprehensive model evaluation
   # - Clinical interpretation of results
   ```

4. **Make predictions for new patients**:
   ```python
   # Example prediction for a high-risk patient
   new_patient = pd.DataFrame([{
       'age': 71, 'sex': 0, 'BP': 120, 'cholestrol': 564
   }])
   prediction = model.predict(new_patient)
   probability = model.predict_proba(new_patient)
   ```

## Model Performance

The logistic regression model shows good diagnostic performance:

- **Overall Accuracy**: 70.6%
- **Precision (No Disease)**: 74%
- **Precision (Heart Disease)**: 65%
- **Recall (No Disease)**: 78%
- **Recall (Heart Disease)**: 61%
- **F1-Score**: 0.70 (weighted average)

### Confusion Matrix Analysis
```
                   Predicted
Actual    No Disease  Heart Disease
No Disease     31          9
Heart Disease  11         17
```

**Clinical Interpretation:**
- **True Negatives (31)**: Correctly identified healthy patients
- **True Positives (17)**: Correctly identified heart disease cases
- **False Positives (9)**: Healthy patients flagged as at-risk (safer error)
- **False Negatives (11)**: Heart disease cases missed (critical to minimize)

## Medical Insights & Risk Factors

### Patient Demographics
- **Average Age**: 54.4 years (range: 29-77)
- **Gender Distribution**: 67.8% male, 32.2% female
- **Heart Disease Prevalence**: 44.4% of patients have heart disease

### Health Metrics Analysis
- **Blood Pressure**: Average 131 mmHg (normal to high range)
- **Cholesterol**: Average 250 mg/dl (borderline high)
- **Age Factor**: Risk increases with age as expected
- **Gender Factor**: Model accounts for gender-based risk differences

### Clinical Risk Assessment
The model considers all major cardiovascular risk factors:
- **Age**: Progressive risk increase with aging
- **Gender**: Historical gender-based risk patterns
- **Hypertension**: Blood pressure as key indicator
- **Hyperlipidemia**: Cholesterol levels for cardiovascular risk

## Business Recommendations

### For Healthcare Providers:
- **Screening Programs**: Use model for initial risk assessment
- **Preventive Care**: Focus on high-risk patient identification
- **Resource Allocation**: Prioritize patients with multiple risk factors
- **Early Intervention**: Implement lifestyle counseling for at-risk patients

### For Public Health:
- **Community Screening**: Deploy model for population health assessment
- **Risk Education**: Target high-risk demographics with health education
- **Prevention Programs**: Focus on modifiable risk factors (BP, cholesterol)

### For Insurance/Healthcare Systems:
- **Risk Stratification**: Support actuarial modeling and premium calculation
- **Care Management**: Identify patients needing intensive monitoring
- **Cost Prediction**: Estimate healthcare resource needs

## Visualizations

The analysis includes essential medical visualizations:

1. **Heart Disease Distribution by Gender**: Count plot showing gender-based disease patterns
2. **Age Distribution**: Histogram of patient ages in the dataset
3. **Risk Factor Correlations**: Relationships between health metrics
4. **Model Performance**: Confusion matrix and classification metrics

## Data Quality

**Exceptional Data Quality:**
- **Complete Dataset**: No missing values across all 270 records
- **Balanced Features**: Good representation across age groups and health metrics
- **Clean Encoding**: All variables properly formatted for analysis
- **Realistic Ranges**: All health metrics within clinically expected ranges

## Statistical Summary

| Health Metric | Mean | Std | Normal Range | Clinical Significance |
|---------------|------|-----|--------------|----------------------|
| **Age** | 54.4 years | 9.1 | Adult population | Risk increases with age |
| **Blood Pressure** | 131 mmHg | 17.9 | <120 (normal) | Borderline high average |
| **Cholesterol** | 250 mg/dl | 51.7 | <200 (desirable) | Above recommended level |
| **Heart Disease Rate** | 44.4% | - | Varies by population | High-risk study population |

## Model Applications

**Clinical Decision Support:**
- Initial screening tool for heart disease risk
- Complement to physician clinical judgment
- Population health risk assessment
- Preventive care program targeting

**Research Applications:**
- Cardiovascular epidemiology studies
- Risk factor analysis and validation
- Healthcare outcome prediction
- Clinical trial patient stratification

## Key Medical Findings

### High-Risk Profile Identification
The model helps identify patients with elevated cardiovascular risk:
- **Older patients** with **multiple risk factors**
- **Elevated blood pressure** combined with **high cholesterol**
- **Gender-specific risk patterns** based on historical medical data

### Preventive Healthcare Insights
- **Blood Pressure Management**: Critical modifiable risk factor
- **Cholesterol Control**: Important for cardiovascular health
- **Age Awareness**: Risk counseling for older patients
- **Gender Considerations**: Tailored risk assessment approaches

## ⚠ Model Limitations & Medical Disclaimers

**Clinical Limitations:**
- **Not a Diagnostic Tool**: Should supplement, not replace, clinical judgment
- **Limited Features**: Missing key factors like family history, lifestyle, ECG results
- **Population Specific**: Based on specific patient population characteristics
- **Moderate Accuracy**: 70.6% accuracy requires clinical validation

**Missing Clinical Factors:**
- Family history of cardiovascular disease
- Smoking status and lifestyle factors
- Exercise capacity and fitness level
- Additional laboratory values (HDL, LDL, triglycerides)
- Imaging and diagnostic test results

**Medical Disclaimer:**
This model is for educational and research purposes only. Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.

## Future Enhancements

**Model Improvements:**
- **Feature Engineering**: Add BMI calculation, risk scores
- **Advanced Algorithms**: Explore Random Forest, SVM, Neural Networks
- **Cross-Validation**: Implement medical-grade validation protocols
- **Ensemble Methods**: Combine multiple algorithms for robust predictions

**Additional Medical Features:**
- **Lifestyle Factors**: Smoking, exercise, diet quality
- **Laboratory Values**: Complete lipid panel, glucose, inflammatory markers
- **Family History**: Genetic predisposition factors
- **Medications**: Current cardiovascular medications
- **Comorbidities**: Diabetes, obesity, other health conditions

## Contributing

Contributions welcome! Priority areas:
- Medical feature engineering
- Advanced classification algorithms
- Clinical validation studies
- Healthcare outcome analysis tools

## Medical Context

**Cardiovascular Disease Overview:**
- Leading cause of death globally
- Major risk factors: age, gender, hypertension, hyperlipidemia
- Preventable through lifestyle modifications and medical management
- Early detection significantly improves outcomes

**Clinical Significance:**
- **Screening Tool**: Supports systematic cardiovascular risk assessment
- **Population Health**: Enables large-scale risk stratification
- **Resource Planning**: Helps healthcare systems allocate preventive care resources

## Technical Notes

**Data Characteristics:**
- **Sample Size**: 270 patients (appropriate for initial modeling)
- **Feature Quality**: Clean, complete medical data
- **Target Balance**: 44.4% positive cases (good for binary classification)
- **No Preprocessing Required**: Data ready for immediate analysis

**Model Configuration:**
- Algorithm: Logistic Regression (suitable for medical binary classification)
- Train/test split: 75/25 with random_state=42
- No feature scaling applied (all features on similar scales)
- Default hyperparameters (no regularization tuning)

**Example Clinical Prediction:**
```
Patient Profile:
- 71-year-old female
- Blood Pressure: 120 mmHg (normal)
- Cholesterol: 564 mg/dl (very high)
Prediction: Heart Disease Present
Clinical Action: Immediate cardiovascular evaluation recommended
```

## Clinical Interpretation Guide

**Risk Factor Thresholds:**
- **Blood Pressure**: <120 (normal), 120-139 (elevated), ≥140 (high)
- **Cholesterol**: <200 (desirable), 200-239 (borderline), ≥240 (high)
- **Age**: Risk increases progressively, especially after 45 (men) and 55 (women)

**Model Output Interpretation:**
- **Probability Score**: Use predict_proba() for risk percentage
- **Binary Classification**: 0 = Low risk, 1 = High risk for heart disease
- **Clinical Context**: Always integrate with comprehensive medical assessment

---

**Model Accuracy**: 70.6% | **Last Updated**: August 2025 | **Medical Disclaimer**: For educational purposes only - consult healthcare professionals for medical decisions
