# Telco Customer Churn

Predicting which customers are likely to churn, using the IBM Telco Customer Churn dataset (~7,000 customers, 20 features across demographics, account info, and subscribed services).

## Project structure

| File | Purpose |
|---|---|
| `Teleco_Customer_Churn.csv` | Raw dataset (place here before running anything) |
| `eda_preprocessing.ipynb` | EDA and preprocessing — cleans the data, runs statistical tests (Mann-Whitney, chi², VIF, Mutual Information) to decide which features to keep, exports `teleco_model_ready.csv` |
| `model_comparison.ipynb` | Trains and compares 4 models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost), tunes the decision threshold for F2, runs 5-fold CV, exports `best_model.pkl` |
| `app.py` | Streamlit demo app — loads `best_model.pkl` and predicts churn risk for a customer.

## Setup


pip install -r requirements.txt



## How to run

Run in this order — each step depends on the file produced by the previous one:

1. **Place the dataset**: put `Teleco_Customer_Churn.csv` in the project root.
2. **Run `eda_preprocessing.ipynb`** Produces `teleco_clean.csv` and `teleco_model_ready.csv`.
3. **Run `model_comparison.ipynb`** Loads `teleco_model_ready.csv`, compares models, and produces `best_model.pkl`.
4. **Launch the demo app**:
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`. Fill in a customer profile and get a churn prediction.

## Results

Best model: **Logistic Regression**, threshold tuned for F2 (recall-weighted) instead of the default 0.5, since missing a churner costs more than a wasted retention offer.

| Model | AUC-ROC | F2 (tuned) | Recall |
|---|---|---|---|
| Logistic Regression | 0.838 | 0.755 | 0.920 |
| Gradient Boosting | 0.840 | 0.754 | 0.936 |
| XGBoost | 0.811 | 0.730 | 0.906 |
| Random Forest | 0.809 | 0.729 | 0.909 |

Full reasoning and statistical validation are in the notebooks.
