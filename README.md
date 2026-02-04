# Credit-Scoring-Model
This project focuses on building a professional credit scoring system (Scorecard) based on historical lending data from Lending Club. The objective is to classify customers as “Good” or “Bad” in order to support lending decisions and optimize business strategies.
## Overview
The model uses the Logistic Regression algorithm combined with Weight of Evidence (WoE) and Information Value (IV) techniques—considered gold standards in the banking industry—to ensure transparency and high interpretability.
## Dataset
- **Source:** https://www.kaggle.com/datasets/wordsforthewise/lending-club 
- **Analyzed Variables:**

  **Target Variable:** Created from the loan_status column. A value of 1 (Bad) is assigned to risky loans (Default, Charged Off, Late), while 0 (Good) is assigned to safe loans (Fully Paid).
  **Independent Variables:** Include customers’ financial information such as annual_inc (income), dti (debt-to-income ratio), int_rate (interest rate), loan_amnt (loan amount), home_ownership, emp_length (employment length), and other credit history variables.
- **Access:** The dataset is loaded directly within the Jupyter Notebook.

## Objective
- Build a classification model to identify customers who are likely to repay their loans and those who pose a higher credit risk.
- Establish a credit scorecard ranging from 300 to 850.
- Provide approval cut-offs based on the expected bad rate to support business strategy.

## Step 1: Import Libraries
- Import data:
- Input:
```python
# 1. Basic data processing libraries
import pandas as pd
import numpy as np

# 2. Data visualization libraries (For EDA and Dashboards)
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Machine Learning & Statistics libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# 4. VIF calculation library (To check for multicollinearity)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 5. Display configuration
%matplotlib inline
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.float_format', lambda x: '%.3f' % x) # Format float for readability
```
## Step 2: Load Data & Filtering
- Import data:
- Input:
```python
# 1. Declare list of required columns
use_cols = [
    'loan_status', 'annual_inc', 'loan_amnt', 'term', 'int_rate',
    'emp_length', 'home_ownership', 'dti', 'delinq_2yrs',
    'revol_util', 'bc_util', 'open_acc', 'inq_last_6mths'
]

# 2. Load data (Ensure the file path is correct)
# Note: Use low_memory=False to avoid errors with large datasets
df = pd.read_csv('/content/drive/MyDrive/SWC Project/accepted_2007_to_2018Q4.csv', usecols=use_cols, low_memory=False)

# 3. Filter out "Current" and ambiguous loan statuses
# Keep only completed loans to clearly identify Good or Bad customers
status_to_keep = ['Fully Paid', 'Charged Off', 'Default', 'Late (31-120 days)']
df = df[df['loan_status'].isin(status_to_keep)]

# 4. Create binary Target variable: 1 for Bad (Risk), 0 for Good (Safe)
df['target'] = np.where(df['loan_status'] == 'Fully Paid', 0, 1)

print(f"Total rows after filtering: {df.shape[0]}")
print("Target Distribution:")
print(df['target'].value_counts(normalize=True) * 100)
```

- Output:

Total rows after filtering: 1366817

Target Distribution:

target

0   78.778
1   21.222

Name: proportion, dtype: float64

- Import data:
- Input:
```python
df.info()
df.head()
```

- Output:

**Dataset Shape:** 1,366,817 rows × 14 columns  
**Memory Usage:** 156.4+ MB  

| # | Column           | Non-Null Count | Data Type |
|---|------------------|---------------|-----------|
| 0 | loan_amnt        | 1,366,817     | float64   |
| 1 | term             | 1,366,817     | object    |
| 2 | int_rate         | 1,366,817     | float64   |
| 3 | emp_length       | 1,286,450     | object    |
| 4 | home_ownership   | 1,366,817     | object    |
| 5 | annual_inc       | 1,366,817     | float64   |
| 6 | loan_status      | 1,366,817     | object    |
| 7 | dti              | 1,366,419     | float64   |
| 8 | delinq_2yrs      | 1,366,817     | float64   |
| 9 | inq_last_6mths   | 1,366,816     | float64   |
|10 | open_acc         | 1,366,817     | float64   |
|11 | revol_util       | 1,365,933     | float64   |_

| loan_amnt | term       | int_rate | emp_length | home_ownership | annual_inc | loan_status | dti  | delinq_2yrs | inq_last_6mths | open_acc | revol_util | bc_util | target |
|-----------|------------|----------|------------|----------------|------------|-------------|------|--------------|----------------|-----------|-------------|---------|--------|
| 3600      | 36 months  | 13.99    | 10+ years  | MORTGAGE       | 55000      | Fully Paid  | 5.91 | 0            | 1              | 7         | 29.7        | 37.2    | 0      |
| 24700     | 36 months  | 11.99    | 10+ years  | MORTGAGE       | 65000      | Fully Paid  |16.06 | 1            | 4              | 22        | 19.2        | 27.1    | 0      |
| 20000     | 60 months  | 10.78    | 10+ years  | MORTGAGE       | 63000      | Fully Paid  |10.78 | 0            | 0              | 6         | 56.2        | 55.9    | 0      |
| 10400     | 60 months  | 22.45    | 3 years    | MORTGAGE       | 104433     | Fully Paid  |25.37 | 1            | 3              | 12        | 64.5        | 77.5    | 0      |
| 11950     | 36 months  | 13.44    | 4 years    | RENT           | 34000      | Fully Paid  |10.20 | 0            | 0              | 5         | 68.4        | 91.0    | 0      |


## Step 3: Data Cleaning (Type Casting, Missing & Outliers)
### 3.1 Sanity Check
- Import data:
- Input:
```python
# 1. Missing values statistics table
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
sanity_tab = pd.concat([missing_data, missing_percent], axis=1, keys=['Missing Count', 'Percent (%)'])

print("--- 1. MISSING DATA CHECK ---")
display(sanity_tab.sort_values(by='Percent (%)', ascending=False))

# 2. Data Visualization
plt.figure(figsize=(18, 12))

# --- Chart 1: Target Variable Ratio (Data Imbalance) ---
plt.subplot(2, 2, 1)
sns.countplot(x='target', data=df, palette='viridis')
plt.title('Target Variable Distribution (0: Good, 1: Bad)', fontsize=14)

# --- Chart 2: Missing Values Heatmap (10k samples) ---
plt.subplot(2, 2, 2)
sns.heatmap(df.sample(min(10000, len(df))).isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data Map (Yellow streaks represent nulls)', fontsize=14)

# --- Chart 3: Annual Income Outliers Boxplot ---
plt.subplot(2, 2, 3)
sns.boxplot(x='target', y='annual_inc', data=df, palette='Set2')
plt.title('Outlier Check: Income vs Target (Log Scale)', fontsize=14)
plt.yscale('log')

# --- Chart 4: Interest Rate Distribution (Predictive Power) ---
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='int_rate', hue='target', kde=True, bins=30, palette='magma')
plt.title('Interest Rate Distribution by Good/Bad Group', fontsize=14)

plt.tight_layout()
plt.show()

# 3. Detailed inspection of Categorical variables
print("\n--- 2. CATEGORICAL VARIABLES DETAILS ---")
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    print(f"\nValue distribution for column '{col}':")
    # Note: Displaying top 10 values including NaNs
    print(df[col].value_counts(dropna=False).head(10))
```

-Output:

## 1. Missing Data Check

| Column            | Missing Count | Percent (%) |
|------------------|---------------|-------------|
| emp_length       | 80,367        | 5.880       |
| bc_util          | 62,234        | 4.553       |
| revol_util       | 884           | 0.065       |
| dti              | 398           | 0.029       |
| inq_last_6mths   | 1             | 0.000       |
| loan_amnt        | 0             | 0.000       |
| annual_inc       | 0             | 0.000       |
| home_ownership   | 0             | 0.000       |
| int_rate         | 0             | 0.000       |
| term             | 0             | 0.000       |
| delinq_2yrs      | 0             | 0.000       |
| loan_status      | 0             | 0.000       |
| open_acc         | 0             | 0.000       |
| target           | 0             | 0.000       |

<img width="1790" height="1189" alt="image" src="https://github.com/user-attachments/assets/e61ad072-af53-4d8b-a9db-e3e242a8fc97" />

## 2. Categorical Variables Details

### Term Distribution

| Term       | Count   |
|------------|----------|
| 36 months | 1,033,242 |
| 60 months | 333,575   |

---

### Employment Length Distribution

| Employment Length | Count   |
|-------------------|----------|
| 10+ years        | 448,782  |
| 2 years          | 123,714  |
| < 1 year         | 110,094  |
| 3 years          | 109,396  |
| 1 year           | 89,937   |
| 5 years          | 85,473   |
| 4 years          | 81,971   |
| NaN              | 80,367   |
| 6 years          | 63,654   |
| 8 years          | 61,459   |

---

### Home Ownership Distribution

| Home Ownership | Count   |
|----------------|----------|
| MORTGAGE      | 674,904  |
| RENT          | 543,891  |
| OWN           | 147,526  |
| ANY           | 304      |
| OTHER         | 144      |
| NONE          | 48       |

---

### Loan Status Distribution

| Loan Status           | Count   |
|------------------------|----------|
| Fully Paid            | 1,076,751 |
| Charged Off           | 268,559   |
| Late (31-120 days)    | 21,467    |
| Default               | 40        |

**Exploratory Data Analysis (EDA) Summary**

**Missing Data**
The dataset is relatively clean. The highest missing rates appear in emp_length (5.88%) and bc_util (4.55%), while other key variables show negligible or zero missing values. Overall, data completeness is sufficient to support reliable model development.

**Target Distribution**
The target variable derived from loan_status exhibits a strong class imbalance. Fully Paid (Good) loans dominate the dataset, whereas Charged Off, Late, and Default (Bad) loans represent a smaller proportion. This imbalance reflects real-world credit portfolios and motivates the use of techniques such as class weighting or resampling during model training.

**Loan Characteristics**
Loan Term: 36-month loans are nearly three times more frequent than 60-month loans.
Employment Length: A significant share of borrowers have long employment tenure (10+ years), which is generally associated with lower credit risk.
Home Ownership: Most borrowers either have a mortgage or rent their homes, while outright ownership represents a smaller segment.

**Data Quality Note**
Minor visualization warnings related to plotting palettes were addressed. The observed categorical distributions are consistent with industry-standard credit datasets, confirming the dataset’s suitability for credit scoring and risk modeling.






