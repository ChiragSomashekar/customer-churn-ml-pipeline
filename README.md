# Customer Churn Prediction & Intervention Strategy

A machine learning pipeline built on the Telco dataset to predict which customers are likely to leave, explain why, and recommend what to do about it.

## The Problem

1,869 customers churned in this dataset. That's $121K in monthly revenue lost. The question isn't just who's leaving, it's whether we can predict it early enough to act.

## What I Built

A Logistic Regression model with SHAP explainability and business-driven threshold optimization. The model catches 95% of churners (vs 56% at the default threshold) by treating a missed churner ($2,100 lost CLV) as 21x more costly than a false alarm ($100 wasted offer).

The output isn't just a prediction, it's a ranked list of customers by risk tier, with the top churn drivers for each one and a recommended intervention.

## Results

| Metric | Value |
|--------|-------|
| Model AUC | 84.2% |
| Churners caught | 95% (vs 56% at default threshold) |
| Modelled savings | $658K (test set, assumes 100% intervention success) |

## Risk Tiers

| Tier | Customers | Churn Rate | Action |
|------|-----------|------------|--------|
| Critical | 94 (7%) | 74.5% | Personal call + contract upgrade |
| High | 341 (24%) | 52.5% | Proactive outreach + loyalty discount |
| Medium | 250 (18%) | 28.4% | Email campaign |
| Low | 724 (51%) | 7.5% | Monitor only |

## Tech Stack

Python, pandas, scikit-learn, SHAP, matplotlib

## The Notebook

Full analysis including EDA, feature engineering, modelling, threshold tuning, 
SHAP explainability, and intervention strategy.

[View on GitHub](https://github.com/ChiragSomashekar/customer-churn-ml-pipeline/blob/main/notebooks/churn_analysis.ipynb) | [View on nbviewer](https://nbviewer.org/github/ChiragSomashekar/customer-churn-ml-pipeline/blob/main/notebooks/churn_analysis.ipynb)
