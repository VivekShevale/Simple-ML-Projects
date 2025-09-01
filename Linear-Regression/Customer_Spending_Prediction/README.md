# E-commerce Customer Analysis - Linear Regression Model

A machine learning project that predicts yearly customer spending for an e-commerce company using linear regression. This analysis helps understand which customer behaviors most strongly correlate with higher spending.

## Project Overview

This project analyzes e-commerce customer data to build a predictive model for yearly spending amounts. The model uses customer behavioral metrics to forecast spending patterns, providing valuable insights for business strategy and customer segmentation.

## Objectives

- Predict yearly customer spending based on behavioral metrics
- Identify the most important factors influencing customer spending
- Evaluate model performance using statistical metrics
- Provide actionable insights for business decision-making

## Dataset Description

The dataset contains 500 customer records with the following features:

### Features Used for Prediction
- **Avg. Session Length**: Average duration of in-store style advice sessions
- **Time on App**: Average time spent on the mobile application (minutes)
- **Time on Website**: Average time spent on the website (minutes)
- **Length of Membership**: Number of years as a member

### Target Variable
- **Yearly Amount Spent**: Total amount spent by customer per year (USD)

### Additional Columns (Not Used in Model)
- **Email**: Customer email address
- **Address**: Customer address
- **Avatar**: Customer avatar color

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
├── Ecommerce Customers    # Dataset file
├── linear_regression_analysis.ipynb    # Main analysis script
└── README.md                  # This file
```

## Usage

1. **Load and explore the data**:
   ```python
   import pandas as pd
   ec = pd.read_csv("Ecommerce Customers.csv")
   ec.head()
   ```

2. **Run the complete analysis**:
   ```python
   # The script includes:
   # - Data exploration and visualization
   # - Train/test split (70/30)
   # - Model training
   # - Performance evaluation
   # - Feature importance analysis
   ```

3. **Make predictions for new customers**:
   ```python
   # Example prediction for average customer
   new_customer = pd.DataFrame({
       'Avg. Session Length': [33.05],
       'Time on App': [12.05],
       'Time on Website': [37.06],
       'Length of Membership': [3.53]
   })
   predicted_amount = model.predict(new_customer)
   ```

## Model Performance

The linear regression model demonstrates excellent performance:

- **R² Score**: 0.989 (98.9% of variance explained)
- **Mean Absolute Error (MAE)**: $7.23
- **Root Mean Square Error (RMSE)**: $8.93
- **Mean Square Error (MSE)**: $79.81

## Key Insights

### Feature Importance (Coefficients)
1. **Length of Membership**: 61.28 (Most Important)
   - Each additional year of membership correlates with ~$61 more in yearly spending
2. **Time on App**: 38.59
   - Each additional minute on the app correlates with ~$39 more in yearly spending
3. **Avg. Session Length**: 25.98
   - Longer advice sessions correlate with higher spending
4. **Time on Website**: 0.19 (Least Important)
   - Website time has minimal impact on spending

### Business Recommendations
- **Focus on member retention**: Length of membership is the strongest predictor
- **Invest in mobile app development**: App engagement significantly impacts spending
- **Optimize in-store advice sessions**: Session length positively correlates with spending
- **Website optimization is lower priority**: Minimal impact on yearly spending

## Visualizations

The analysis includes several key visualizations:

1. **Pairplot**: Relationships between features and target variable
2. **Prediction vs Actual**: Scatter plot showing model accuracy
3. **Residual Distribution**: Histogram of prediction errors
4. **Feature Importance**: Bar chart of coefficient values

## Data Quality

- **Complete Dataset**: No missing values across all 500 records
- **Balanced Features**: All numerical features are well-distributed
- **No Outliers**: Data appears clean and consistent

## Statistical Summary

| Metric | Avg. Session Length | Time on App | Time on Website | Length of Membership | Yearly Amount Spent |
|--------|-------------------|-------------|-----------------|-------------------|-------------------|
| **Mean** | 33.05 | 12.05 | 37.06 | 3.53 | $499.31 |
| **Std** | 0.99 | 0.99 | 1.01 | 1.00 | $79.31 |
| **Min** | 29.53 | 8.51 | 33.91 | 0.27 | $256.67 |
| **Max** | 36.14 | 15.13 | 40.01 | 6.92 | $765.52 |

## Model Interpretation

The linear regression equation:
```
Yearly Spending = -1047.93 + 25.98×(Avg Session Length) + 38.59×(Time on App) + 0.19×(Time on Website) + 61.28×(Length of Membership)
```

## Future Enhancements

- Implement cross-validation for more robust evaluation
- Explore polynomial features for non-linear relationships
- Add regularization techniques (Ridge/Lasso regression)
- Incorporate additional customer demographics
- Deploy model as a web service for real-time predictions

## Contributing

Feel free to fork this project and submit pull requests for improvements. Areas for contribution:
- Additional feature engineering
- Alternative modeling approaches
- Enhanced visualizations
- Model deployment scripts

## Notes

- Model assumes linear relationships between features and target
- Results are based on historical data and may not reflect future trends
- Regular model retraining recommended as new data becomes available

---

**Model Accuracy**: 98.9% | **Last Updated**: August 2025
