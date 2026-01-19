
# Customer Analytics: EDA, CLV Prediction & Segmentation

## Project Overview

This project performs **end-to-end customer analytics** on a digital wallet dataset, covering:

1. **Exploratory Data Analysis (EDA)** to understand customer behavior
2. **Feature engineering** for transactional and engagement metrics
3. **Customer Lifetime Value (CLV) prediction** using machine learning
4. **Customer segmentation** based on predicted CLV
5. **Business action recommendations** for each segment

The objective is to demonstrate how raw customer transaction data can be transformed into **actionable business insights**.

---

## Dataset Summary

* **Records**: 7,000 customers
* **Columns**: 20 original features + engineered features
* **Target Variable**:

  * `LTV` (historical)
  * `Future_CLV` (engineered & predicted)

### Key Feature Categories

* **Demographics**: Age, Location, Income Level
* **Transaction Behavior**:

  * Total Transactions
  * Average / Max / Min Transaction Value
  * Total Spent
* **Engagement Metrics**:

  * Active Days
  * App Usage Frequency
  * Loyalty Points
  * Referral Count
* **Service Experience**:

  * Support Tickets Raised
  * Issue Resolution Time
  * Customer Satisfaction Score

### Data Quality Checks

* No missing values
* No duplicate records
* Clean numeric and categorical separation

---

## Exploratory Data Analysis (EDA)

### Key Analyses Performed

* Age distribution using custom age bins
* Customer distribution across locations (Urban, Suburban, Rural)
* Income level vs location analysis
* Transaction and spending distributions
* Loyalty points distribution and top customer identification
* Engagement level classification based on active days
* Spender categorization using quartiles:

  * Low Spender
  * Medium Spender
  * High Spender

### Derived Customer Segments

* **Age_Group**
* **Spender_Category**
* **Engagement_Level**

These segments help understand **who the customers are** and **how they interact with the platform**.

---

## Feature Engineering for CLV

New features were created to improve predictive power:

* `frequency` = Total Transactions
* `monetary` = Total Spent
* `value_freq_interaction` = Avg Transaction Value × Frequency
* `cashback_spend_ratio` = Cashback / Total Spent
* `Retention_Rate` (simulated for future value modeling)
* `Future_CLV` and log-transformed target `Future_CLV_log`

---

## CLV Prediction Model

### Model Used

* **Gradient Boosting Regressor**
* Implemented using a **scikit-learn Pipeline**
* Preprocessing handled via **ColumnTransformer**:

  * StandardScaler for numerical features
  * OneHotEncoder for categorical features

### Model Optimization

* GridSearchCV with 3-fold cross-validation
* Tuned parameters:

  * Number of estimators
  * Max depth
  * Learning rate

### Model Performance

* **RMSE**: ~669,776
* **R² Score**: ~0.95
* **MAPE**: ~11.9%

This indicates strong predictive performance for future CLV estimation.

---

## Customer Segmentation

### Method

* **KMeans clustering** based on:

  * Predicted Future CLV
  * Transaction frequency
  * Monetary value
  * Interaction features

### Final Segments

* **High Value Customers**
* **Medium Value Customers**
* **Low Value Customers**

Segments are ranked based on average predicted CLV.

---

## Business Recommendations

Each customer is assigned a business action:

| Segment      | Recommended Action                      |
| ------------ | --------------------------------------- |
| High Value   | Upsell premium or VIP services          |
| Medium Value | Personalized engagement and nudges      |
| Low Value    | Standard nurturing and retention offers |

---

## Outputs

* Trained CLV model saved using `joblib`
* Final customer insights exported to CSV:

```
outputs/customer_insights_recommendations_final.csv
```

This file includes:

* Predicted CLV
* Customer segment
* Recommended business action

---

## Project Structure

```
customer-clv-analysis/
│
├── notebooks/
│   ├── eda.ipynb
│   └── clv_modeling.ipynb
│
├── models/
│   └── clv_model.pkl
│
├── outputs/
│   └── customer_insights_recommendations_final.csv
│
├── requirements.txt
└── README.md
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* Joblib

---

## Key Takeaways

* EDA reveals strong variation in customer value and engagement
* Feature engineering significantly improves CLV prediction
* Gradient Boosting performs well for tabular customer data
* CLV-based segmentation enables clear, actionable business strategies

---

## Future Improvements

* Replace simulated retention rate with real churn data
* Add SHAP for model explainability
* Deploy as a Streamlit or API-based application
* Integrate real-time prediction for new customers

---

## Author

**Jai Kishan Kokkiligadda**
Data Analyst | Data Science 
* Help fix the **new customer prediction KeyError**
* Convert this into a **production-ready ML pipeline**
