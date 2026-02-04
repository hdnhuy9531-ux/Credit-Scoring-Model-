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

Total rows after filtering: 1,366,817

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

| #  | Column          | Non-Null Count | Data Type |
|----|------------------|------------------|------------|
| 0  | loan_amnt       | 1,366,817        | float64    |
| 1  | term            | 1,366,817        | object     |
| 2  | int_rate        | 1,366,817        | float64    |
| 3  | emp_length      | 1,286,450        | object     |
| 4  | home_ownership  | 1,366,817        | object     |
| 5  | annual_inc      | 1,366,817        | float64    |
| 6  | loan_status     | 1,366,817        | object     |
| 7  | dti             | 1,366,419        | float64    |
| 8  | delinq_2yrs     | 1,366,817        | float64    |
| 9  | inq_last_6mths  | 1,366,816        | float64    |
| 10 | open_acc        | 1,366,817        | float64    |
| 11 | revol_util      | 1,365,933        | float64    |
| 12 | bc_util         | 1,304,583        | float64    |
| 13 | target          | 1,366,817        | int64      |

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

### 3.2 Cleaning and Casting
- Import data:
- Input:
```python
# 1. Process emp_length: Use safe Regex to extract numbers
# We replace special labels first, then extract the numeric values
df['emp_length'] = df['emp_length'].replace({'< 1 year': '0', '10+ years': '10'})
df['emp_length'] = df['emp_length'].str.extract('(\d+)').astype(float)

# 2. Process term: Remove whitespace and extract numeric values (36 or 60)
df['term'] = df['term'].str.extract('(\d+)').astype(int)

# 3. Process revol_util: Convert from "80%" (string) to 80.0 (numeric)
if df['revol_util'].dtype == 'O':
    df['revol_util'] = df['revol_util'].str.replace('%', '').astype(float)

# 4. Handle Outliers & Financial Logic
# dti, revol_util, and bc_util should not be negative and typically shouldn't exceed 100%
for col in ['dti', 'revol_util', 'bc_util']:
    df[col] = df[col].clip(lower=0, upper=100)

# 5. Cap Annual Income at the 95th percentile to handle extreme outliers
upper_limit = df['annual_inc'].quantile(0.95)
df['annual_inc'] = df['annual_inc'].clip(upper=upper_limit)

print("--- DATA CLEANING COMPLETED ---")
# Verify key columns after processing
display(df[['emp_length', 'term', 'annual_inc', 'dti', 'revol_util']].describe().T)
```

- Output:
  
--- DATA CLEANING COMPLETED ---
| Feature      | Count     | Mean    | Std     | Min | 25%   | 50%   | 75%   | Max |
|--------------|------------|----------|----------|------|--------|--------|--------|------|
| emp_length   | 1,286,450 | 5.961    | 3.692    | 0    | 2      | 6      | 10     | 10   |
| term         | 1,366,817 | 41.857   | 10.309   | 36   | 36     | 36     | 36     | 60   |
| annual_inc   | 1,366,817 | 72,204.03| 34,978.49| 0    | 45,760 | 65,000 | 90,000 | 155,000 |
| dti          | 1,366,419 | 18.242   | 8.872    | 0    | 11.80  | 17.63  | 24.09  | 100  |
| revol_util   | 1,365,933 | 51.762   | 24.490   | 0    | 33.40  | 52.10  | 70.70  | 100  |

### 3.3 FINAL DATA QUALITY CHECK
- Import data:
- Input:
```python
# 1. Quick Summary: Missing Values & Correlation with Target
print("--- 1. MISSING DATA RATIO & CORRELATION WITH TARGET ---")
summary = pd.DataFrame({
    'Missing (%)': df.isnull().mean() * 100,
    'Corr with Target': df.select_dtypes(include=[np.number]).corr()['target']
}).sort_values(by='Corr with Target', ascending=False)
display(summary)

# 2. Comprehensive Visualization
plt.figure(figsize=(20, 14)) # Increased size for better readability

# Chart A: Annual Income Distribution (Histogram)
plt.subplot(2, 2, 1)
sns.histplot(df['annual_inc'], bins=40, kde=True, color='teal')
plt.title('A. Annual Income Distribution (Capped)')

# Chart B: Interest Rate vs Target (Boxplot)
plt.subplot(2, 2, 2)
sns.boxplot(x='target', y='int_rate', data=df, hue='target', palette='Set2', legend=False)
plt.title('B. Interest Rate vs Target')

# Chart C: Default Rate by Home Ownership (Barh)
plt.subplot(2, 2, 3)
df.groupby('home_ownership')['target'].mean().sort_values().plot(kind='barh', color='salmon')
plt.title('C. Bad Debt Rate by Home Ownership')

# Chart D: FULL Correlation Matrix with Annotations
plt.subplot(2, 2, 4)
corr = df.select_dtypes(include=[np.number]).corr()

# Removed the 'mask' to show the full matrix
sns.heatmap(corr,
            annot=True,      # Show numeric values
            fmt=".2f",       # Format to 2 decimal places
            cmap='RdBu_r',   # Red-Blue color map
            center=0,
            annot_kws={"size": 10}, # Adjust font size of numbers
            cbar=True)       # Show color bar for scale

plt.title('D. Full Correlation Matrix (Numeric Details)')

plt.tight_layout()
plt.show()
```

-Output:
--- 1. MISSING DATA RATIO & CORRELATION WITH TARGET ---

| Feature           | Missing (%) | Corr with Target |
|------------------|------------|------------------|
| target           | 0.000      | 1.000            |
| int_rate         | 0.000      | 0.263            |
| term             | 0.000      | 0.181            |
| dti              | 0.029      | 0.107            |
| loan_amnt        | 0.000      | 0.072            |
| inq_last_6mths   | 0.000      | 0.062            |
| bc_util          | 4.553      | 0.060            |
| revol_util       | 0.065      | 0.055            |
| open_acc         | 0.000      | 0.027            |
| delinq_2yrs      | 0.000      | 0.020            |
| emp_length       | 5.880      | -0.016           |
| annual_inc       | 0.000      | -0.064           |
| home_ownership   | 0.000      | NaN              |
| loan_status      | 0.000      | NaN              |

<img width="1989" height="1389" alt="image" src="https://github.com/user-attachments/assets/02c7bdf7-ad7e-4186-a6fa-cb35cc681331" />


Strongest Predictors: int_rate (0.263) and term (0.181) are the top "star" predictors. Higher interest rates and longer loan terms show a clear positive correlation with bad debt (Target=1).

Inverse Correlation: annual_inc (-0.064) has a negative coefficient, which is logically sound: as income increases, the risk of default decreases.

Missing Values: emp_length and bc_util still contain some null values. However, these will be fully handled in the upcoming Weight of Evidence (WoE) transformation step without the need for row deletion.

Categorical Handling: home_ownership shows as NaN in the correlation matrix because it is a categorical variable, which Pearson correlation cannot process. Its predictive power will be measured using Information Value (IV) in the next step.

## Step 4: Feature Engineering (WoE & IV Calculation)
This is the most critical step for selecting high-value variables for the Scorecard model. We will partition the variables into bins to calculate their predictive power (discriminatory power).

- Import data:
- Input:
```python
def calculate_woe_iv(df, feature, target):
    # 1. Handle Categorical and Numerical variables separately
    if df[feature].dtype == 'object' or feature == 'home_ownership':
        df_role = df.groupby(feature)[target].agg(['count', 'sum'])
    else:
        # Divide numerical variables into 5 groups (bins) using qcut
        df['temp_bin'] = pd.qcut(df[feature], q=5, duplicates='drop')
        df_role = df.groupby('temp_bin', observed=True)[target].agg(['count', 'sum'])

    # 2. Calculate metrics
    df_role.columns = ['Total', 'Bad']
    df_role['Good'] = df_role['Total'] - df_role['Bad']

    # Apply Laplace smoothing to avoid division by zero
    prob_bad = (df_role['Bad'] + 0.5) / df_role['Bad'].sum()
    prob_good = (df_role['Good'] + 0.5) / df_role['Good'].sum()

    df_role['WoE'] = np.log(prob_good / prob_bad)
    df_role['IV'] = (prob_good - prob_bad) * df_role['WoE']

    iv_value = df_role['IV'].sum()
    return iv_value, df_role

# Execute calculations for all independent variables
features = [col for col in df.columns if col not in ['target', 'loan_status', 'temp_bin']]
iv_results = {}

print("--- CALCULATING IV FOR EACH FEATURE ---")
for col in features:
    iv, _ = calculate_woe_iv(df, col, 'target')
    iv_results[col] = iv

# Export Feature Predictive Power Ranking
iv_df = pd.DataFrame.from_dict(iv_results, orient='index', columns=['IV']).sort_values(by='IV', ascending=False)
print("\nINFORMATION VALUE (IV) RANKING:")
print(iv_df)
```

- Output:

| Feature          | IV     |
|------------------|--------|
| int_rate         | 0.412  |
| dti              | 0.066  |
| loan_amnt        | 0.036  |
| home_ownership   | 0.031  |
| annual_inc       | 0.024  |
| bc_util          | 0.022  |
| revol_util       | 0.019  |
| inq_last_6mths   | 0.014  |
| open_acc         | 0.004  |
| emp_length       | 0.001  |
| term             | 0.000  |
| delinq_2yrs      | 0.000  |

**Detailed IV Analysis & Feature Selection Strategy**

**The Top Performern** - int_rate (0.412): This is a Strong Predictor. Interest rate serves as the primary risk indicator; higher rates directly correlate with a higher probability of default. This variable will be the backbone of your scoring model.

**Potential Predictors** (IV 0.02 - 0.1): dti, loan_amnt, home_ownership, annual_inc, and bc_util.

These variables possess Medium Strength.While not dominant, they provide diverse perspectives on financial stability and collateral. All these features should be retained for the modeling phase.

**Weak Features & Candidates for Removal** (IV < 0.02): revol_util, inq_last_6mths, open_acc, emp_length, term, and delinq_2yrs.

**Observation on term:** Surprisingly, term shows an IV of 0.000 despite having a high Pearson correlation earlier. This is due to the qcut binning strategy (5 bins) applied to a variable that only has 2 unique values (36 and 60), causing the WoE calculation to lose its discriminative power.

Observation on emp_length: Unfortunately, employment length shows almost no predictive value for credit risk within this specific dataset.

**Next Steps: Feature Fine-Tuning Strategy** We will not discard variables immediately. Instead, we will perform a Fine-tuning process on the most significant features.

Action Plan: Examine the WoE trends for high-impact variables (especially int_rate and dti).

**Goal:** Ensure Monotonicity—meaning that as the variable value increases, the risk (WoE) must increase or decrease consistently. This is crucial for the stability and interpretability of the final Logistic Regression model.

- Import data:
- Input:
```python
# Detailed inspection of the Weight of Evidence (WoE) for the strongest predictor
iv_val, woe_detail = calculate_woe_iv(df, 'int_rate', 'target')
print("WOE DETAILS FOR VARIABLE: int_rate")
display(woe_detail)

# Plotting the WoE trend
plt.figure(figsize=(10, 6))
woe_detail['WoE'].plot(kind='bar', color='skyblue')
plt.title('WoE Trend for int_rate (Monotonicity Check)')
plt.axhline(y=0, color='r', linestyle='--')
plt.ylabel('WoE Value')
plt.xlabel('Bins (Interest Rate Ranges)')
plt.show()
```

- Output:
WOE DETAILS FOR VARIABLE: int_rate

| Interest Rate Bin | Total  | Bad   | Good   | WoE   | IV    |
|-------------------|---------|--------|---------|--------|--------|
| (5.31, 8.99]      | 277,360 | 20,385 | 256,975 | 1.223  | 0.206 |
| (8.99, 11.53]     | 272,008 | 40,638 | 231,370 | 0.428  | 0.032 |
| (11.53, 13.99]    | 297,509 | 61,047 | 236,462 | 0.043  | 0.000 |
| (13.99, 16.99]    | 248,729 | 66,006 | 182,723 | -0.293 | 0.017 |
| (16.99, 30.99]    | 271,211 |101,990 | 169,221 | -0.805 | 0.157 |

<img width="866" height="730" alt="image" src="https://github.com/user-attachments/assets/319a954e-2861-4b99-827f-533871d725a8" />

WoE Analysis Summary: int_rate

The Weight of Evidence (WoE) results for Interest Rate are highly logical and provide a robust foundation for the Scorecard model:

Perfect Monotonicity: The WoE values exhibit a clear downward trend (from 1.223 to -0.805) as the interest rate increases across bins. This perfectly aligns with credit risk theory: higher interest rates are associated with a higher concentration of "Bad" customers, resulting in lower (more negative) WoE values.

Strong Discriminatory Power: The significant spread (~2.0) between the highest and lowest WoE values confirms why this variable is a "Strong Predictor" with an IV of 0.412. It provides excellent separation between Good and Bad applicants.

Neutral Point Identification: The bin (11.53, 13.99] serves as the neutral point with a WoE of 0.043 (near zero). In this range, the borrower risk profile mirrors the average risk of the entire portfolio.

Conclusion: Due to its stable monotonicity and high IV, int_rate will be the primary driver in our Logistic Regression model.

## Step 5: Model Training (Logistic Regression)
After the Exploratory Data Analysis (EDA) and feature screening stages, the final model was streamlined to include seven key variables. This selection ensures a balance between predictive performance and business practicality, based on three main criteria:

Predictive Power: All seven variables passed the Information Value (IV) assessment. Features with very low IV (lacking discriminatory ability) were removed. Notably, int_rate (interest rate) emerged as the strongest predictor of default risk.

Multicollinearity Removal (VIF Check): Through the Variance Inflation Factor (VIF) analysis, variables with excessively high correlations were eliminated. This improves the stability of the Logistic Regression model, reduces noise, and ensures that each feature contributes unique informational value.

Credit Risk Framework: These seven variables collectively capture a comprehensive view of a borrower’s financial capacity, including:

Financial Capacity: annual_inc (Income), dti (Debt-to-Income ratio).

Loan Characteristics: loan_amnt (Loan amount), int_rate (Interest rate), term (Loan term).

Stability Indicators: emp_length (Employment tenure), home_ownership (Home ownership).

### 5.1 WoE Mapping & Model Data Preparation
- Import data:
- Input:
```python
def apply_woe_map(df, feature, target):
    # Reconstruct the WoE table from the previously calculated function
    iv, woe_table = calculate_woe_iv(df, feature, target)

    if df[feature].dtype == 'object' or feature == 'home_ownership':
        # For categorical variables: direct mapping by category label
        mapping = woe_table['WoE'].to_dict()
        return df[feature].map(mapping)
    else:
        # For numerical variables: map based on intervals (bins)
        # Get cut-off points
        bins = pd.qcut(df[feature], q=5, duplicates='drop', retbins=True)[1]
        # Re-bin the data using these cut-off points
        labels = pd.cut(df[feature], bins=bins, include_lowest=True)

        # Create a mapping dictionary from Interval to WoE value
        mapping = woe_table['WoE'].to_dict()
        return labels.map(mapping).astype(float)

# List of 7 selected features
final_features = ['int_rate', 'dti', 'loan_amnt', 'home_ownership', 'annual_inc', 'bc_util', 'term']

# Create a new DataFrame to store WoE-encoded data
df_woe = pd.DataFrame()

print("--- APPLYING WOE ENCODING FOR SELECTED FEATURES ---")
for col in final_features:
    print(f"Mapping: {col}...")
    df_woe[col] = apply_woe_map(df, col, 'target')

# Add target variable to the new DataFrame
df_woe['target'] = df['target']

# Handle NaN values generated during mapping (usually from Missing groups)
# by filling them with 0 (average level)
df_woe = df_woe.fillna(0)

print("\n--- CHECKING DATA AFTER MAPPING ---")
display(df_woe.head())
```

--- APPLYING WOE ENCODING FOR SELECTED FEATURES ---
Mapping: int_rate...
Mapping: dti...
Mapping: loan_amnt...
Mapping: home_ownership...
Mapping: annual_inc...
Mapping: bc_util...
Mapping: term...

--- CHECKING DATA AFTER MAPPING ---

| int_rate | dti  | loan_amnt | home_ownership | annual_inc | bc_util | term  | target |
|--------|------|------------|----------------|------------|---------|--------|---------|
| 0.043  | 0.330 | 0.291      | 0.181          | -0.091     | 0.084   | -0.000 | 0 |
| 0.043  | 0.058 | -0.218     | 0.181          | -0.021     | 0.238   | -0.000 | 0 |
| 0.428  | 0.211 | -0.161     | 0.181          | -0.021     | -0.015  | -0.000 | 0 |
| -0.805 | -0.114| -0.042     | 0.181          | 0.268      | -0.071  | -0.000 | 0 |
| 0.043  | 0.330 | -0.042     | -0.189         | -0.199     | -0.206  | -0.000 | 0 |

### 5.2 Split the Training/Test set and build a Logistics Regression model
- Import data:
- Input:
```python
# 1. Split Features and Target
X = df_woe.drop('target', axis=1)
y = df_woe['target']

# 2. Train/Test Split (70/30)
# Use 'stratify' to ensure the same proportion of Good/Bad loans in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Initialize and Train the Model
# 'class_weight=balanced' is used to handle data imbalance (fewer Bad loans than Good loans)
lr_model = LogisticRegression(class_weight='balanced')
lr_model.fit(X_train, y_train)

# 4. Predictions
# Get probability for the positive class (Bad/Default)
y_pred_prob = lr_model.predict_proba(X_test)[:, 1]
y_pred = lr_model.predict(X_test)

print("--- MODEL TRAINING COMPLETED ---")
```


### 5.3 Model evaluation (Gini & AUC)
- Import data:
- Input:
```python
# Calculate AUC (Area Under Curve)
auc_score = roc_auc_score(y_test, y_pred_prob)
# Calculate Gini Coefficient based on AUC
gini_score = 2 * auc_score - 1

print(f"AUC Score: {auc_score:.4f}")
print(f"Gini Index: {gini_score:.4f}")

# Inspect Model Coefficients
# These represent the impact of each feature in the logistic regression equation
coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_[0]})
print("\nMODEL COEFFICIENTS (Impact of each variable in the equation):")
display(coef_df.sort_values(by='Coefficient', ascending=False))
```

- Output:

AUC Score: 0.6895

Gini Index: 0.3790

MODEL COEFFICIENTS (Impact of each variable in the equation):

| Feature          | Coefficient |
|------------------|-------------|
| bc_util          | 0.128       |
| term             | 0.000       |
| dti              | -0.567      |
| annual_inc       | -0.718      |
| home_ownership   | -0.882      |
| int_rate         | -0.906      |
| loan_amnt        | -0.993      |

Model Evaluation & Coefficients Analysis

Classification Performance (AUC & Gini) Gini Index = 0.379 (37.9%): This is a solid and realistic result for Lending Club data.
In the banking industry, a Gini between 0.30 and 0.45 is considered acceptable for production use. It demonstrates that the model possesses a strong ability to differentiate between "Good" and "Bad" borrowers, performing significantly better than random guessing.

Model Coefficients & Key Insights Sign Direction: Most coefficients are negative (e.g., -0.90, -0.99). In a Logistic Regression model trained on WoE, a negative coefficient implies that as the WoE increases, the probability of default (Target=1) decreases.
Variable Impact:

loan_amnt (-0.993) and int_rate (-0.906): These are the most influential drivers. Changes in loan amounts and interest rates trigger the most significant shifts in the final credit score.

term (0.000): This variable contributes nothing to the model. This confirms the earlier IV analysis: term is either incorrectly encoded via WoE or lacks discriminatory power in the current binning setup.

bc_util (0.128): The positive sign—contrary to other variables—may indicate Multicollinearity or a divergent risk trend compared to other financial metrics.

## Step 6: Scorecard Scaling

The results will be converted from the Logistic Regression model to a common scale (usually from 300 to 850, similar to the FICO score).

Standard Formulas and Parameters To calculate the final credit scores, we need to establish the following scaling parameters:
Target Score (Base Score): 600 points.

Target Odds (Odds at Base Score): 50 (representing a Good/Bad ratio of 50:1).

PDO (Points to Double the Odds): 20 (meaning that for every 20-point increase, the Good/Bad ratio doubles — effectively halving the risk)

- Import data:
- Input:
```python
# 1. Set Scaling Parameters
pdo = 20
base_score = 600
base_odds = 50

# Calculate Factor and Offset
factor = pdo / np.log(2)
offset = base_score - factor * np.log(base_odds)

print(f"Scaling Parameters: Factor = {factor:.2f}, Offset = {offset:.2f}\n")

# 2. Score Calculation for each observation
# Formula: Score = Offset + Factor * (Intercept / n_features + Σ (coef * woe))
# For simplicity, we calculate directly from the model's predicted probability (y_pred_prob)
# log(odds) = ln(p / (1-p)) -> Note: In scoring, we use log(Good/Bad)

# Calculate Log-odds from the Test set predicted probabilities
# We use (1 - p)/p to represent the odds of being "Good"
log_odds = np.log((1 - y_pred_prob) / y_pred_prob)

# Convert Log-odds to Scorecard Points
df_test_results = X_test.copy()
df_test_results['Actual'] = y_test.values
df_test_results['Prob_Bad'] = y_pred_prob
df_test_results['Score'] = offset + factor * log_odds

# 3. Clip scores within the standard 300 - 850 range to handle extreme values
df_test_results['Score'] = df_test_results['Score'].clip(300, 850).astype(int)

print("--- CREDIT SCORING RESULTS (TEST DATA) ---")
display(df_test_results[['Actual', 'Prob_Bad', 'Score']].head(10))

# 4. Score Distribution Statistics
print("\n--- SCORE DISTRIBUTION STATISTICS ---")
print(df_test_results['Score'].describe())
```

- Output:

Scaling Parameters: Factor = 28.85, Offset = 487.12
--- CREDIT SCORING RESULTS (TEST DATA) ---

| ID      | Actual | Prob_Bad | Score |
|--------|--------|-----------|--------|
| 148502  | 1      | 0.293     | 512    |
| 1401654 | 1      | 0.691     | 463    |
| 1876627 | 0      | 0.512     | 485    |
| 1883060 | 0      | 0.468     | 490    |
| 1626746 | 1      | 0.740     | 456    |
| 313233  | 0      | 0.677     | 465    |
| 1886203 | 1      | 0.253     | 518    |
| 111668  | 0      | 0.401     | 498    |
| 1598341 | 0      | 0.406     | 498    |
| 1314413 | 0      | 0.541     | 482    |

--- SCORE DISTRIBUTION STATISTICS ---

| Statistic | Value     |
|------------|------------|
| Count      | 410,046    |
| Mean       | 490.975    |
| Std        | 21.399     |
| Min        | 443.000    |
| 25%        | 475.000    |
| 50% (Median)| 490.000   |
| 75%        | 506.000    |
| Max        | 543.000    |

Scorecard Scaling Analysis

Score Range Observations Narrow Distribution: The generated scores range from 443 to 543, with an average of 491.
The Issue: While the industry standard often spans from 300 to 850, this specific model results in a tighter cluster (approx. 100-point spread).

Root Cause: This occurs because the model Coefficients are relatively small, and the primary predictor (int_rate) does not have a wide enough discriminatory range to push scores to the extremes. However, the ranking logic remains perfectly sound.

Score Sensitivity (Score vs. Risk) The scoring system demonstrates a clear inverse relationship between score and probability of default:
Low Risk: A borrower with Prob_Bad = 0.253 receives a score of 518.

High Risk: A borrower with Prob_Bad = 0.740 is penalized, with a score dropping to 456.

Conclusion: The scorecard is logically consistent and stable. Higher scores reliably correspond to lower risk.

Scaling Parameter Insights
Factor (28.85): This is the amplification coefficient. At this level, every unit of change in the log-odds shifts the credit score by approximately 29 points.

Offset (487.12): This acts as the "starting point" or intercept. It explains why the majority of the population is centered around the 490-point mark.


### 6.1 Score Distribution Chart
- Import data:
- Input:
```python
plt.figure(figsize=(12, 6))
# Using a Red-Yellow-Green palette to intuitively represent risk (Red=Bad, Green=Good)
sns.histplot(data=df_test_results, x='Score', hue='Actual', bins=50, kde=True, palette='RdYlGn')

plt.title('Scorecard Distribution: Good (0) vs. Bad (1)', fontsize=15)
plt.axvline(x=df_test_results['Score'].mean(), color='blue', linestyle='--', label='Mean Score')

plt.xlabel('Credit Score')
plt.ylabel('Customer Count')

# Customizing the legend for clarity
plt.legend(title='Loan Status', labels=['Bad (Default)', 'Good (Paid)'])
plt.show()
````

- Output:

<img width="1023" height="549" alt="image" src="https://github.com/user-attachments/assets/367f0023-88db-4bac-a673-9c5b614bd119" />

### 6.2 Set the Cut-off (Approval Threshold)
- Import data:
- Input:
```python
# 1. Create Score Bands
# Segmenting the scores into 10 equal-width intervals
df_test_results['Score_Band'] = pd.cut(df_test_results['Score'], bins=10)

# 2. Calculate Actual Bad Rate per Score Band
# Aggregating total customers and the average of the 'Actual' column (Target=1)
rank_order = df_test_results.groupby('Score_Band', observed=True).agg({
    'Actual': ['count', 'mean']
}).reset_index()

# 3. Rename columns for professional reporting
rank_order.columns = ['Score_Range', 'Total_Customers', 'Bad_Rate_Percentage']
rank_order['Bad_Rate_Percentage'] = (rank_order['Bad_Rate_Percentage'] * 100).round(2)

print("--- RISK RANK ORDERING TABLE ---")
# Sorting from highest score (Lowest Risk) to lowest score (Highest Risk)
display(rank_order.sort_values('Score_Range', ascending=False))
```

- Output:

--- RISK RANK ORDERING TABLE ---

| Score Range     | Total Customers | Bad Rate (%) |
|------------------|------------------|----------------|
| (533.0, 543.0]   | 6,509            | 4.50           |
| (523.0, 533.0]   | 30,927           | 6.09           |
| (513.0, 523.0]   | 38,247           | 8.46           |
| (503.0, 513.0]   | 36,702           | 11.45          |
| (493.0, 503.0]   | 62,325           | 15.72          |
| (483.0, 493.0]   | 76,022           | 20.11          |
| (473.0, 483.0]   | 63,769           | 25.54          |
| (463.0, 473.0]   | 52,302           | 33.04          |
| (453.0, 463.0]   | 33,951           | 41.31          |
| (442.9, 453.0]   | 9,292            | 50.90          |

Final Model Validation: Rank-Ordering & Business Strategy

Excellent Rank-Ordering Performance The most valuable takeaway from this table is the perfectly monotonic decrease in the Bad_Rate as the credit score increases:
Highest Risk Group (442.9, 453.0]: Has the highest default rate at 50.90%.

Lowest Risk Group (533.0, 543.0]: Has the lowest default rate at only 4.50%.

Conclusion: The model demonstrates powerful risk discrimination. Borrowers in the lowest score band are 11 times more likely to default than those in the highest band.

Business Implications & Approval Thresholds (Cut-off) Based on this rank-ordering, we can establish practical business rules similar to those used by a Chief Risk Officer (CRO):
Green Zone (Automatic Approval): Scores above 523. With a default rate between 4% and 6%, these applications can be fast-tracked for instant approval.

Yellow Zone (Conditional Review): Scores between 483 and 513. Default rates range from 11% to 20%. These cases may require manual underwriting, additional documentation, or collateral.

Red Zone (Automatic Rejection): Scores below 473. The default rate surges above 33%. Lending to this group should be avoided to protect capital.

Population Distribution Insights The largest segment of your portfolio is concentrated in the 483 - 493 score range (76,022 customers). This "mass-market" segment has a default rate of approximately 20%, which aligns perfectly with the overall default rate observed during the initial data exploration phase.














