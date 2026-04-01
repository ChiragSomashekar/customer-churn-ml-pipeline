# Customer Churn Prediction & Intervention Strategy

A machine learning pipeline that predicts customer churn and provides actionable intervention recommendations for business teams.

## Business Problem

Customer churn costs ~$121,000/month in lost revenue. This project answers:
- **Who** is likely to leave?
- **Why** are they leaving?
- **What** should we do about it?

## Results

| Metric | Value |
|--------|-------|
| Model AUC | 84% |
| Churners Identified | 95% (up from 56% at default threshold) |
| Potential Savings | $658,200 (test set) |

### Top Churn Drivers
1. New customers (0-12 months) - 48% churn rate
2. Fiber optic service - 42% churn rate
3. Month-to-month contracts - 43% churn rate

## Approach

### 1. Business-Driven Threshold Optimization
Instead of using the default 0.5 threshold, I optimized for business costs:
- Cost of missing a churner (False Negative): $2,100 (lost CLV)
- Cost of unnecessary intervention (False Positive): $100

**Result:** Optimal threshold of 0.1 catches 95% of churners vs 56% at default.

### 2. SHAP Explainability
Every prediction is explainable. Business teams receive:
- Churn probability
- Risk tier (Critical/High/Medium/Low)
- Top 3 personalized risk factors per customer
- Recommended action

### 3. Tiered Intervention Strategy

| Risk Tier | % of Customers | Actual Churn Rate | Action |
|-----------|----------------|-------------------|--------|
| Critical | 7% | 74.5% | Personal call + contract upgrade |
| High | 24% | 52.5% | Proactive outreach + loyalty discount |
| Medium | 18% | 28.4% | Email campaign |
| Low | 51% | 7.5% | No intervention |

## Tech Stack

- **Python** (pandas, numpy, matplotlib, seaborn)
- **Modeling:** scikit-learn (Logistic Regression), XGBoost
- **Explainability:** SHAP
- **Data:** Telco Customer Churn dataset 

## Notebook & Code

Access the full end-to-end analysis, including preprocessing, modeling, and SHAP explainability:

- 👉 [View on GitHub](https://github.com/ChiragSomashekar/customer-churn-ml-pipeline/blob/main/notebooks/churn_analysis.ipynb)  
- 👉 [View Clean Version (nbviewer)](https://nbviewer.org/github/ChiragSomashekar/customer-churn-ml-pipeline/blob/main/notebooks/churn_analysis.ipynb)


## Key Learnings

- **Threshold tuning matters:** Default thresholds optimize for accuracy, not business outcomes
- **Explainability builds trust:** SHAP analysis enables stakeholders to understand and act on predictions
- **Cost asymmetry drives strategy:** When missing a churner costs 21x more than a wasted offer, you should flag more customers

## Future Improvements

1. Wrap in sklearn Pipeline for cleaner deployment
2. A/B test intervention strategies
3. Build automated weekly scoring pipeline
4. Monitor for model drift

## Author

Chirag Somashekar
