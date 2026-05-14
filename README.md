# End-to-End Customer Churn Analysis and Prediction

A comprehensive machine learning pipeline for predicting customer churn in a telecom dataset. Combines unsupervised clustering and supervised classification to identify at-risk customers and quantify revenue at stake — so businesses can act before customers leave.

---

## Business Output

- `churn_predictions_final.csv` — churn risk score and probability for every customer
- `churn_risk_by_segment.csv` — revenue at risk and recommended retention strategies by customer segment

---

## Dataset

**Telco Customer Churn with Realistic Feedback**
Source: [Kaggle](https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback?select=telco_churn_with_all_feedback.csv)

7,000+ customer records covering demographics, service subscriptions, monthly/total charges, and simulated customer feedback.

---

## Methodology

### 1. Data Preparation
- Explored dataset dimensions, data types, and missing values
- Fixed `TotalCharges` type conversion and imputed missing values
- Removed irrelevant `PromptInput` column
- EDA: churn distribution, categorical feature breakdowns, tenure and charge trends

### 2. Feature Engineering & Preprocessing
- One-hot encoding for binary and nominal categorical features
- Ordinal encoding for `Contract` type
- Standard scaling for numerical features
- Sentiment scores and topic labels added as new features

### 3. Unsupervised Learning
- **Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment` transformer + Claude API for detailed feedback scoring
- **Topic Classification**: Keyword-based classification into business topics (e.g., Price Competitiveness, Service Quality)
- **PCA**: Reduced feature space to 2 dimensions for visualisation
- **K-Means Clustering**: k=3 selected via Elbow Method; Claude API used to generate plain-English segment labels from cluster statistics

### 4. Supervised Learning: Churn Prediction
- **Model**: Random Forest Classifier
- **Split**: 80/20 train-test, stratified
- **Feature leakage**: `sentiment_score_nlp` detected and removed due to spurious correlation with the target
- **Threshold tuning**: Optimal threshold set to 0.3 (down from default 0.5) to prioritise recall and minimise total business cost

---

## Key Results

**Model Performance (threshold = 0.3)**

| Metric | Value |
|---|---|
| Training Accuracy | 82.34% |
| Test Accuracy | 77% |
| Churn Recall | 78.9% |
| Improvement over Baseline | +6.60% |

**Business Impact**

| Metric | Value |
|---|---|
| Total Customers Analysed | 7,043 |
| Customers Flagged At-Risk | 1,934 (27.5%) |
| Total Revenue at Risk | $2,108,239.10 |
| Retention Budget Needed | $67,690.00 |
| Net Revenue at Risk | $2,040,549.10 |
| Cost Saving vs. Default Threshold | $216,215.85 |

---

## Customer Segments

| Segment | Customers | Net Revenue at Risk | Retention ROI |
|---|---|---|---|
| High-Risk Mid-Tenure (Cluster 0) | 1,763 | $1,405,244.40 | ~2,277% |
| Premium Loyal (Cluster 1) | — | Highest value per customer | Critical |
| Budget-Conscious Satisfied (Cluster 2) | — | Lowest churn rate | Stable |

---

## Recommendations

1. **Prioritise High-Risk Mid-Tenure customers** for immediate retention campaigns
2. **Tailor offers by segment:**
   - High-Risk: contract upgrades, service quality improvements
   - Premium Loyal: VIP treatment, loyalty rewards
   - Budget-Conscious: price-lock guarantees, value-driven plans
3. **Deploy at threshold 0.3** to maximise churn prevention and minimise business cost
4. **Retrain quarterly** to adapt to shifting customer behaviour

---

## Tools & Stack

| Category | Tool |
|---|---|
| Language | Python |
| Notebook | Google Colab |
| ML | scikit-learn (Random Forest, K-Means, PCA) |
| NLP | HuggingFace Transformers (RoBERTa) |
| AI | Claude API (sentiment detail, cluster labelling) |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |

---

## How to Run

1. Open `churn_analysis_project.ipynb` in Google Colab
2. Upload the dataset from [Kaggle](https://www.kaggle.com/datasets/beatafaron/telco-customer-churn-realistic-customer-feedback?select=telco_churn_with_all_feedback.csv)
3. Run all cells in order

---

*Built with Python, scikit-learn, HuggingFace Transformers, and Claude API.*
