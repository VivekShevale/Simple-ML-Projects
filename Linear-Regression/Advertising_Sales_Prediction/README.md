# Advertising Sales Prediction - Linear Regression Model

A machine learning project that predicts product sales based on advertising spend across different media channels (TV, Radio, and Newspaper). This analysis helps optimize advertising budgets and understand the effectiveness of different marketing channels.

## Project Overview

This project analyzes advertising data to build a predictive model for sales revenue. The model uses advertising expenditure across three channels to forecast sales performance, providing valuable insights for marketing budget allocation and ROI optimization.

## Objectives

- Predict sales revenue based on advertising spend across different channels
- Identify which advertising channels have the highest impact on sales
- Evaluate model performance and accuracy
- Provide data-driven recommendations for advertising budget allocation

## Dataset Description

The dataset contains 200 advertising campaigns with the following features:

### Features Used for Prediction
- **TV**: Advertising budget spent on TV (thousands of dollars)
- **Radio**: Advertising budget spent on Radio (thousands of dollars)  
- **Newspaper**: Advertising budget spent on Newspaper (thousands of dollars)

### Target Variable
- **Sales**: Product sales in response to advertising (thousands of units)

### Additional Columns
- **Unnamed: 0**: Index column (excluded from analysis)

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
├── Advertising.csv           # Dataset file
├── advertising_analysis.ipynb   # Main analysis script
└── README.md                # This file
```

## Usage

1. **Load and explore the data**:
   ```python
   import pandas as pd
   ad = pd.read_csv("Advertising.csv")
   ad.head()
   ```

2. **Run the complete analysis**:
   ```python
   # The script includes:
   # - Data exploration and visualization
   # - Train/test split (80/20)
   # - Model training with Linear Regression
   # - Performance evaluation
   # - Feature importance analysis
   ```

3. **Make predictions for new advertising campaigns**:
   ```python
   # Example prediction
   new_campaign = pd.DataFrame({
       'TV': [150],
       'Radio': [30], 
       'Newspaper': [20]
   })
   predicted_sales = model.predict(new_campaign)
   ```

## Model Performance

The linear regression model shows strong predictive performance:

- **R² Score**: 0.899 (89.9% of variance explained)
- **Mean Absolute Error (MAE)**: 1.46 thousand units
- **Root Mean Square Error (RMSE)**: 1.78 thousand units
- **Mean Square Error (MSE)**: 3.17

## Key Insights

### Feature Importance (Coefficients)
1. **Radio**: 0.189 (Most Effective)
   - Each $1K spent on radio advertising increases sales by ~189 units
   - Highest ROI among all channels
2. **TV**: 0.045 (Moderate Effectiveness)
   - Each $1K spent on TV advertising increases sales by ~45 units
   - Steady, reliable channel for sales growth
3. **Newspaper**: 0.003 (Least Effective)
   - Minimal impact on sales (only ~3 units per $1K spent)
   - Lowest ROI among all channels

### Business Recommendations

**Budget Allocation Strategy:**
- **Prioritize Radio Advertising**: Highest coefficient suggests best ROI
- **Maintain TV Presence**: Solid contributor to sales growth
- **Minimize Newspaper Spend**: Very low impact on sales performance

**Marketing Mix Optimization:**
- Radio advertising appears to be the most cost-effective channel
- TV advertising provides consistent, moderate returns
- Consider reallocating newspaper budget to radio or TV channels

## Visualizations

The analysis includes comprehensive visualizations:

1. **Pairplot Analysis**: Relationship between each advertising channel and sales
2. **Prediction Accuracy**: Scatter plot of actual vs predicted sales
3. **Residual Analysis**: Distribution of prediction errors
4. **Feature Importance Chart**: Horizontal bar chart showing coefficient values

## Data Quality

- **Complete Dataset**: No missing values across all 200 records
- **Clean Data**: All numerical features are properly formatted
- **Balanced Distribution**: Features show good statistical properties

## Statistical Summary

| Metric | TV | Radio | Newspaper | Sales |
|--------|-------|--------|-----------|--------|
| **Mean** | $147.04K | $23.26K | $30.55K | 14.02K units |
| **Std** | $85.85K | $14.85K | $21.78K | 5.22K units |
| **Min** | $0.70K | $0.00K | $0.30K | 1.60K units |
| **Max** | $296.40K | $49.60K | $114.00K | 27.00K units |

## Model Interpretation

The linear regression equation:
```
Sales = 2.98 + 0.045×(TV) + 0.189×(Radio) + 0.003×(Newspaper)
```

**Interpretation:**
- Base sales of ~2,980 units without advertising
- Radio advertising is ~4x more effective than TV
- Newspaper advertising has minimal impact
- Model explains 89.9% of sales variance

## Key Business Insights

1. **Radio is King**: Radio advertising shows the highest return on investment
2. **TV is Steady**: TV provides consistent, predictable sales increases
3. **Newspaper is Ineffective**: Very low correlation with sales performance
4. **Channel Synergy**: Combined approach with emphasis on radio and TV optimal

## Future Enhancements

- **Advanced Models**: Explore Random Forest, XGBoost for non-linear relationships
- **Cross-Validation**: Implement k-fold validation for robust evaluation
- **Interaction Effects**: Analyze synergies between advertising channels
- **Time Series Analysis**: Incorporate seasonal and temporal effects
- **Real-time Dashboard**: Deploy model for live advertising optimization

## Contributing

Contributions welcome! Areas for improvement:
- Feature engineering (interaction terms, transformations)
- Alternative modeling approaches
- Enhanced visualization techniques
- A/B testing framework integration

## ⚠ Limitations

- Model assumes linear relationships between advertising spend and sales
- Does not account for external factors (seasonality, competition, economic conditions)
- Based on historical data - market conditions may change
- Assumes immediate sales response to advertising (no lag effects)

## Model Deployment

For production use:
1. Regular model retraining with new data
2. Monitor for model drift and performance degradation
3. Validate predictions against actual campaign results
4. Consider ensemble methods for improved accuracy

---

**Model Accuracy**: 89.9% | **Last Updated**: August 2025 | **ROI Leader**: Radio Advertising
