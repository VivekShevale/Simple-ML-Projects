# Medical Insurance Cost Prediction - Linear Regression Model

A machine learning project that predicts individual medical insurance charges based on personal and demographic factors. This analysis helps understand the key drivers of healthcare costs and assists in insurance premium estimation.

## Project Overview

This project analyzes medical insurance data to build a predictive model for individual healthcare charges. The model uses personal characteristics including age, BMI, smoking status, and family size to forecast medical costs, providing valuable insights for insurance companies and healthcare planning.

## Objectives

- Predict individual medical insurance charges based on personal characteristics
- Identify the most significant factors driving healthcare costs
- Evaluate model performance for insurance premium estimation
- Provide insights for risk assessment and pricing strategies

## Dataset Description

The medical insurance dataset contains 1,338 individual records with the following features:

### Features Used for Prediction

**Demographic Information:**
- **age**: Age of the primary beneficiary (18-64 years)
- **sex**: Gender of the insurance contractor (encoded: 0=female, 1=male)
- **region**: Beneficiary's residential area (encoded: 0-3 for different regions)

**Health & Lifestyle Factors:**
- **bmi**: Body Mass Index (kg/m²) - measure of body fat based on height and weight
- **smoker**: Smoking status (encoded: 0=no, 1=yes)

**Family Information:**
- **children**: Number of children/dependents covered by insurance (0-5)

### Target Variable
- **charges**: Individual medical costs billed by health insurance (USD)

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
├── insurance.csv              # Dataset file
├── model_train.ipynb   # Main model notebook
└── README.md                  # This file
```

## Usage

1. **Load and explore the data**:
   ```python
   import pandas as pd
   ins = pd.read_csv("insurance.csv")
   ins.head()
   ```

2. **Data preprocessing**:
   ```python
   # Encode categorical variables
   from sklearn.preprocessing import LabelEncoder
   cols = ['sex', 'smoker', 'region']
   for col in cols:
       le = LabelEncoder()
       ins[col] = le.fit_transform(ins[col])
   ```

3. **Run the complete analysis**:
   ```python
   # The script includes:
   # - Data exploration and categorical encoding
   # - Exploratory data analysis with visualizations
   # - Train/test split (70/30)
   # - Model training and evaluation
   # - Feature importance analysis
   ```

4. **Make predictions for new patients**:
   ```python
   # Example prediction using average values
   new_patient = pd.DataFrame([X.mean().values], columns=X.columns)
   predicted_cost = model.predict(new_patient)
   ```

## Model Performance

The linear regression model demonstrates solid predictive performance:

- **R² Score**: 0.750 (75.0% of variance explained)
- **Mean Absolute Error (MAE)**: $4,449
- **Root Mean Square Error (RMSE)**: $6,417
- **Mean Square Error (MSE)**: $41,183,311

## Key Insights

### Feature Importance (Coefficients)

**Most Significant Cost Drivers:**
1. **Smoker** (23,461): **SMOKING IS THE DOMINANT FACTOR**
   - Smoking status increases medical costs by ~$23,461
   - By far the most impactful health factor
2. **Children** (529): Each additional child increases costs by ~$529
3. **BMI** (319): Each BMI unit increase adds ~$319 in medical costs
4. **Age** (237): Each additional year adds ~$237 in medical costs

**Factors with Negative/Minimal Impact:**
- **Region** (-457): Regional differences reduce costs by ~$457
- **Sex** (-39): Gender has minimal impact on medical costs

### Critical Health Insights

**Smoking Impact:**
- Smoking is the overwhelming cost driver (>10x any other factor)
- Smokers pay approximately $23,461 more in medical costs annually
- This represents the single most important modifiable risk factor

**Lifestyle Factors:**
- BMI significantly impacts costs - obesity prevention programs could reduce expenses
- Age-related costs are predictable and gradual (+$237/year)
- Family size affects costs but at a manageable rate (+$529/child)

## Business Recommendations

### For Insurance Companies:
- **Risk-Based Pricing**: Implement significant premium adjustments for smokers
- **Wellness Programs**: Invest in smoking cessation and weight management programs
- **Preventive Care**: Focus on early intervention for high-BMI individuals
- **Family Plans**: Adjust pricing models based on dependent count

### For Healthcare Policy:
- **Smoking Prevention**: Prioritize anti-smoking campaigns and support programs
- **Obesity Prevention**: Implement community health and nutrition programs
- **Regional Analysis**: Investigate regional cost differences for targeted interventions

### For Individuals:
- **Quit Smoking**: Most impactful decision for reducing medical costs
- **Maintain Healthy BMI**: Weight management significantly affects healthcare expenses
- **Preventive Care**: Regular health monitoring to catch issues early

## Visualizations

The analysis includes comprehensive visualizations:

1. **Missing Data Heatmap**: Data quality assessment
2. **Pairplot Analysis**: Relationships between BMI, age, and medical charges
3. **Prediction Accuracy**: Scatter plot showing model performance
4. **Residual Distribution**: Error analysis for model validation
5. **Feature Importance Chart**: Clear visualization of cost drivers

## Data Quality

**Perfect Dataset:**
- **No Missing Values**: Complete data across all 1,338 records
- **Balanced Demographics**: Good representation across age groups and regions
- **Clean Encoding**: Categorical variables properly encoded for analysis

**Encoding Details:**
- **Sex**: 0=Female, 1=Male
- **Smoker**: 0=No, 1=Yes
- **Region**: 0-3 representing different geographical areas

## Statistical Summary

| Feature | Mean | Std | Min | Max | Impact on Cost |
|---------|------|-----|-----|-----|----------------|
| **age** | 39.2 years | 14.0 | 18 | 64 | +$237/year |
| **bmi** | 30.7 kg/m² | 6.1 | 16.0 | 53.1 | +$319/unit |
| **children** | 1.1 | 1.2 | 0 | 5 | +$529/child |
| **smoker** | 20.5% | - | 0 | 1 | +$23,461 |
| **charges** | $13,270 | $12,110 | $1,122 | $63,770 | Target |

## Model Interpretation

The linear regression equation:
```
Medical Charges = -10,710 + 237×(age) + (-39)×(sex) + 319×(bmi) + 
                  529×(children) + 23,461×(smoker) + (-457)×(region)
```

**Key Interpretations:**
- **Base Cost**: ~$10,710 baseline medical expenses
- **Smoking Premium**: Smokers pay an additional $23,461 annually
- **Age Factor**: Costs increase by $237 per year of age
- **Weight Impact**: Each BMI point adds $319 in annual costs
- **Family Size**: Each child adds $529 to family medical expenses

## Critical Health Economics Findings

### The Smoking Cost Crisis
- **Catastrophic Impact**: Smoking increases costs by 177% ($23,461 vs average $13,270)
- **Public Health Priority**: Smoking cessation programs offer massive cost savings
- **Insurance Implications**: Smoker vs non-smoker pricing differentials are justified

### BMI and Healthcare Costs
- **Obesity Factor**: Higher BMI directly correlates with increased medical expenses
- **Prevention Value**: Weight management programs have clear ROI
- **Gradual Impact**: Each BMI unit adds $319 annually

### Age and Family Factors
- **Predictable Aging**: Steady $237/year increase with age
- **Family Planning**: Each child adds $529 in medical costs
- **Life Stage Planning**: Costs are predictable across life stages

## ⚠ Model Limitations

- **Linear Assumptions**: May not capture complex health interactions
- **Missing Factors**: Doesn't include pre-existing conditions, genetics, lifestyle details
- **Regional Simplification**: Limited regional granularity
- **Temporal Factors**: No consideration of healthcare inflation or policy changes
- **Moderate Accuracy**: 75% R² suggests additional factors influence costs

## Future Enhancements

**Model Improvements:**
- **Feature Engineering**: Create interaction terms (age×smoker, bmi×age)
- **Advanced Algorithms**: Explore Random Forest, Gradient Boosting
- **Cross-Validation**: Implement robust validation strategies
- **Regularization**: Apply Ridge/Lasso for feature selection

**Additional Data Sources:**
- **Medical History**: Pre-existing conditions, chronic diseases
- **Lifestyle Factors**: Exercise habits, diet quality, stress levels
- **Genetic Factors**: Family medical history, genetic predispositions
- **Environmental Data**: Air quality, access to healthcare facilities

## Contributing

Contributions welcome! Priority areas:
- Advanced feature engineering techniques
- Ensemble modeling approaches
- Healthcare economics analysis
- Interactive cost calculators

## Use Cases

**Insurance Industry:**
- Premium calculation and risk assessment
- Underwriting decision support
- Actuarial modeling and forecasting

**Healthcare Planning:**
- Resource allocation and capacity planning
- Public health intervention prioritization
- Cost-effectiveness analysis of prevention programs

**Personal Finance:**
- Healthcare budgeting and planning
- Insurance plan comparison and selection
- Lifestyle choice impact assessment

## Technical Notes

**Preprocessing Steps:**
- Label encoding for categorical variables (sex, smoker, region)
- No missing values to handle
- No feature scaling applied (Linear Regression handles different scales)
- Train/test split: 70/30 with random_state=43

**Model Configuration:**
- Algorithm: Ordinary Least Squares Linear Regression
- All 6 features included in final model
- No regularization applied

**Data Distribution:**
- **Age Range**: 18-64 years (working age population)
- **BMI Range**: 15.96-53.13 kg/m² (underweight to severely obese)
- **Cost Range**: $1,122-$63,770 (wide variation in medical expenses)

---

**Model Accuracy**: 75.0% | **Last Updated**: August 2025 | **Key Finding**: Smoking increases costs by $23,461 annually
