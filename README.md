# 📉 From Data to Detection: Building a Robust Financial Fraud Prediction Model


## 🚀 Project Overview
Fraud detection is a classic "needle in a haystack" problem. Legitimate transactions vastly outnumber fraudulent ones, which can hide fraud from basic machine learning models[cite: 4]. A model that misses fraud (low recall) results in direct financial loss and damages customer trust[cite: 4]. 

This project aims to build a robust classification model that effectively identifies fraudulent transactions while minimizing false alarms, transitioning from a heavily imbalanced dataset to a highly sensitive fraud-detection engine[cite: 4].

## 🛠️ Tech Stack & Libraries
*   **Data Manipulation & Math:** `pandas`, `numpy`[cite: 5]
*   **Machine Learning:** `scikit-learn` (`LogisticRegression`, `DecisionTreeClassifier`), `statsmodels.api` (`Logit`)[cite: 5]
*   **Imbalanced Data Handling:** `imblearn` (`SMOTE`)[cite: 5]
*   **Model Explainability:** `shap` (`LinearExplainer`)[cite: 5]
*   **Visualization:** `matplotlib.pyplot`[cite: 5]

## 📊 The Dataset & Preprocessing
The project utilizes a synthetic dataset of 50,000 transactions[cite: 4, 5]. 
*   **Initial Cleaning:** Dropped non-predictive columns (`Transaction_ID`, `User_ID`, `Timestamp`, `Location`)[cite: 5]. 
*   **Encoding:** Applied `LabelEncoder` to categorical features including `Transaction_Type`, `Device_Type`, `Merchant_Category`, `Card_Type`, and `Authentication_Method`[cite: 5].
*   **Class Imbalance:** Discovered a ~2:1 imbalance (33,933 legitimate vs. 16,067 fraudulent transactions)[cite: 5].

## 🧠 Methodology & Machine Learning Pipeline

### 1. Baseline Model (Statsmodels Logit)
We performed an 80/20 train-test split and trained a baseline Logistic Regression model using `statsmodels.api.Logit`[cite: 5].
*   **Baseline Test Metrics:** AUC: 0.8939 | Precision: 0.7107 | Recall: 0.6349 | F1-Score: 0.6707[cite: 5].
*   **The Problem:** A Recall of 63.49% meant the model failed to catch over 36% of actual fraud, representing 1,181 false negatives (missed frauds) in the test set alone[cite: 4, 5]. 

### 2. Data Balancing (SMOTE)
To tackle the model's bias towards the majority class, we applied **Synthetic Minority Over-sampling Technique (SMOTE)** to the training data[cite: 4, 5].
*   **Result:** The training set was balanced from an original shape of (40000, 16) to (54336, 16), ensuring an exact 50/50 split of 27,168 instances per class[cite: 5]. 

### 3. Feature Selection (Forward Stepwise Regression)
To remove noise and improve accuracy, we wrote a custom **Forward Stepwise Selection** algorithm utilizing a p-value threshold of `0.05`[cite: 4, 5]. 
*   **Selected Features:** 12 optimal features were retained out of the original 16, including `Risk_Score`, `Failed_Transaction_Count_7d`, `Is_Weekend`, `Device_Type`, `Transaction_Type`, `Authentication_Method`, `Card_Type`, `Previous_Fraudulent_Activity`, `Merchant_Category`, `IP_Address_Flag`, `Daily_Transaction_Count`, and `Transaction_Amount`[cite: 5].
*   **Statistically Insignificant:** Features like `Account_Balance` and `Transaction_Distance` did not significantly contribute to fraud detection and were dropped[cite: 4].

### 4. Final Model & Evaluation
We trained the final Logistic Regression model on the SMOTE-balanced data using only the selected features[cite: 5].
*   **Final Test Metrics:** AUC: 0.8826 | Precision: 0.6405 | Recall: 0.7657 | F1-Score: 0.6975[cite: 5].
*   **Robustness Check:** 5-Fold Cross-Validation on the resampled data yielded a highly stable mean AUC of **0.8971** (std: 0.0130)[cite: 5].

### 5. Alternative View (Decision Tree)
To provide simple, human-readable business rules, we trained a `DecisionTreeClassifier` configured with `max_depth=6` and `min_samples_leaf=100`[cite: 5].
*   **Example Rule:** A transaction with a `Risk_Score > 0.65` AND `Failed_Transaction_Count_7d > 1.5` is highly likely to be fraudulent[cite: 4].

## 💡 Key Learnings & Recommendations
*   **Success:** We successfully improved the model's ability to catch fraud by increasing Recall from ~63% to **~77%**[cite: 4, 5].
*   **Feature Drivers:** Utilizing `shap.LinearExplainer`, we identified that `Risk_Score`, `Failed_Transaction_Count_7d`, and `Is_Weekend` were the most impactful predictors of fraud[cite: 4, 5].
*   **Business Action:** We recommend deploying the tuned Logistic Regression model for real-time scoring, while utilizing the Decision Tree's logic to explain "flagged" transactions to non-technical stakeholders[cite: 4].
