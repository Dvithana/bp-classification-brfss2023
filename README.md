# High Blood Pressure Classification — BRFSS 2023

Can demographic, lifestyle and health survey data flag people who report high blood pressure?
This project builds and compares machine learning classifiers on the CDC's **BRFSS 2023** survey (433,323 respondents, 350 variables).

> Data Science module coursework — BSc (Hons) Computing, Coventry University (via NIBM).
> This is a **screening-support** study, not a medical diagnostic tool.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dvithana/bp-classification-brfss2023/blob/main/DS_CW.ipynb)

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 72.74% | 68.36% | 64.02% | 66.12% | 0.7998 |
| **XGBoost (best)** | **72.97%** | 67.91% | **66.28%** | **67.08%** | **0.8024** |

Logistic Regression was used as the baseline.

**Key findings**

- **Threshold tuning:** lowering the XGBoost decision threshold from 0.50 to 0.40 raised recall from **66.3% to 78.0%**, so it caught more real cases at the cost of more false positives.
- **Feature ablation (ROC-AUC):** demographic + socioeconomic only **0.736**, + lifestyle **0.770**, full feature set **0.800**. Lifestyle and health features add real predictive value.
- **EDA:** high blood pressure rises with age, from 8.7% (18–24) to 60.8% (65+). Respondents who reported no exercise had 52.4% prevalence vs 38.0% for those who exercised. These are associations, not causes.
- Strongest correlations with high blood pressure (Spearman): age (ρ = 0.38), high cholesterol (ρ = 0.29), BMI (ρ = 0.23).

<!-- Add a chart: save a figure from the notebook into images/ and uncomment -->
<!-- ![Model comparison](images/model_comparison.png) -->

## Approach

1. **Target:** built from `bphigh6`. Kept only clear Yes/No answers and removed borderline, pregnancy-only, refused and missing answers, which left **424,077** valid records (58.45% no / 41.55% yes).
2. **Cleaning:** converted survey codes such as 7, 9, 77, 99, 777 and 999 ("Don't know" / "Refused") to missing values. Rescaled BMI (stored ×100). Converted alcohol from days per week or per month to one drinking-days-per-month value. Kept plausible BMI outliers (IQR check).
3. **Features:** 13 predictors covering age group, sex, BMI, cholesterol, diabetes, exercise, smoking, alcohol, income, education, employment, marital status and routine check-ups.
4. **Pipeline:** a scikit-learn `Pipeline` with median imputation (numeric), most-frequent imputation and one-hot encoding (categorical). This prevents data leakage.
5. **Split:** 80/20 stratified, `random_state=42`.
6. **Models:** Logistic Regression (`max_iter=1000`), Random Forest (100 trees, max depth 15, min leaf 5), XGBoost (200 trees, max depth 6, learning rate 0.1).
7. **Metrics:** accuracy, precision, recall, F1 and ROC-AUC, with a focus on recall and F1 because a missed case matters most in screening.

## How to run

1. Download the **2023 BRFSS** data from the CDC: <https://www.cdc.gov/brfss/annual_data/annual_2023.html>. The dataset is not included in this repo because of its size.
2. Open the notebook in Google Colab (badge above) or locally.
3. Set the data path in the first cell to where you saved the file.
4. Run all cells. Local setup: `pip install -r requirements.txt`

## Limitations and next steps

- BRFSS is **self-reported**, so some answers may be wrong or missing.
- Hyperparameters were fixed (no full search) because of the dataset size and compute limits.
- Next steps: formal hyperparameter tuning, validation on another dataset or survey year, and checking fairness across population groups.

## Tech

Python · pandas · NumPy · scikit-learn · XGBoost · Matplotlib · Google Colab

## References

- CDC (2025) *2023 BRFSS Survey Data and Documentation.*
- World Health Organization (2025) *Hypertension fact sheet.*
- Chen, T. & Guestrin, C. (2016) XGBoost: A scalable tree boosting system. *KDD '16.*
- Breiman, L. (2001) Random forests. *Machine Learning*, 45, 5–32.
- Pedregosa, F. et al. (2011) Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
