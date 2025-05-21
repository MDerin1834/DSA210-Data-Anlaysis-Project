# Colorectal Cancer Survival Analysis - DSA210 Project

## Stage I - Project Proposal 

### 1. Project Overview
   
   This project aims to analyze a large global dataset on colorectal cancer (CRC) with the goal of developing a machine learning model that can predict survival outcomes based on various factors. The public dataset contains 26 parameters across over 160,000 records, including demographic and medical data related to colorectal cancer patients from various countries taken from kaggle website. 

   Given the size and complexity of the dataset, the project will focus on a manageable subset of 20,000 records for analysis. The subset will be randomly selected. Additionally, the project will incorporate augmented data from external sources, such as age distribution scores and cancer mortality rates (ASR) by country, to enhance the model's predictive performance. This combination of clinical data and contextual country-based information is expected to provide more accurate and insightful predictions.

### 2. Motivation
   
   Colorectal cancer is one of the leading causes of cancer-related deaths worldwide. Since colorectal cancer accounted for 12.7% of all new cancer diagnoses and 12.4% of all cancer-related deaths (ENCR, 2021), I believe the outcome of this project will serve as a valuable tool in raising awareness about the dangers of colorectal cancer. 
   
   Another motivation for this project stems from the need to better understand how factors like age, geographical location, physical features, alcohol or tobacco consupmtion and other demographic information affect colorectal cancer survival rates, thus enabling more tailored healthcare strategies for different regions. Eventually, the project aims to drive insightful information by analyzing and elaborating on colorectal cancer data and various statistics. 

### 3. Project Goals and Objectives
   
   The main goal of this project is to understand the most crucial factors leading to death from colorectal cancer. 
   
   Hypothesis: Colorectal cancer patients in the American continents have lower survival rates compared to other continents due to factors such as less developed healthcare systems and less healthy lifestyles in some American countries as these factors may act as significant barriers to survival. 
   
   Further analysis will explore the leading causes of death from colorectal cancer and identify crucial factors that contribute to its development.

### 4. Data Collection & Enrichment
   
The project will use the following datasets:

Colorectal Cancer Dataset:
As briefly mentioned before this dataset contains 26 parameters (e.g., age, gender, stage of cancer, country, survival outcome) across over 160,000 records of colorectal cancer patients. The parameters are a mix of categorical and numerical variables that describe clinical, demographic, and medical factors influencing colorectal cancer prognosis. Since the data is publicly available and open to being overly used, the enrishment of the data from different sources may be required. The data will be downloaded as .csv file from kaggle website and then will be enriched by adding two score columns as age distribution and deaths by country. These two recent scores will be provided from reliable official websites (Please check the sourses). The data augmentation is planned as following:

Age distribution score: Calculated using data from the United Nations, this score categorizes the global population into age groups, which will be assigned to corresponding patients in the dataset. The estimated distrubution of the new cases of colorectal cancer with respect to age is the following: 150 cases among 0 to 19 years old, 6,880 cases among 20 to 44 years old, 138,722 cases among 45 to 69 years old, and 195,667 cases 70+ years old. These numbers will be standardized and will be added to the data to enrich the data. Age distribution scores (ADS) are listed in corresponding order: 0.00044 (age: 0-19), 0.02015 (age: 20-44), 0.40631(age: 45-69), 0.57310 (Age: 70+).

Death rates by country (ASR): This score, based on country-specific mortality rates, will enhance the model's ability to account for the impact of healthcare systems and country-level factors on colorectal cancer survival. It is planned to used these scores and implement them in my data using reported country information for each student, which I believe will play a significant role in decision-making performance of the ML model. Sources: 

[Mustafa Derin 32272]

Sources:

1- https://encr.eu/sites/default/files/inline-files/Colorectal_cancer_factsheet_March_2021.pdf [ENCR] 

2- https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions

3- Wang, C. C., Sung, W. W., Yan, P. Y., Ko, P. Y., & Tsai, M. C. (2021). Favorable colorectal cancer mortality-to-incidence ratios in countries with high expenditures on health and development index: A study based on GLOBOCAN database. Medicine, 100(41), e27414. https://doi.org/10.1097/MD.0000000000027414

4- https://www.wcrf.org/preventing-cancer/cancer-statistics/colorectal-cancer-statistics/ [WCRF] 


## Stage II - Exploratory Data Analysis (EDA) & Hypothesis Testing

### Data Enrichment & Preprocessing

#### 1.1 Feature Engineering

* Age Distribution Score: 
Created a categorical score based on age brackets and their epidemiological weights:
Rationale: Reflects increased colorectal cancer risk with age (aligned with WHO guidelines).

* Country ASR (Age-Standardized Rate) Score:
Mapped countries to their published ASR values (e.g., Japan: 36.6, India: 4.9) to quantify regional risk.
Handling Missing Data: Filled unmapped countries (e.g., Canada) with 0 (interpreted as "no data").

#### 1.2 Data Integrity Checks

* Null Values: Confirmed no missing values across all 28 original columns.
Data Types:
6 numerical (e.g., Age, Tumor_Size_mm, Healthcare_Costs).
22 categorical (e.g., Cancer_Stage, Genetic_Mutation).

* Descriptive Statistics:
Mean age: 69.1 years (SD: 11.9), tumor size: 42.0mm (SD: 21.7).
Healthcare costs ranged from $25K to $120K (mean: $72.3K).

### EDA and Visualizations

#### 2.1 Univariate Analysis

* Key Distributions:
Age: Right-skewed, peak at 60–75 years (typical for CRC diagnoses).
Tumor Size: Uniform distribution (5–79mm), no skew.
Mortality: 40% deceased (Mortality=Yes).

* Categorical Variables:
Gender: Male-dominated (60% of cases).
Cancer Stage: Balanced between Localized (40%) and Regional (40%), but fewer Metastatic (20%).
Early Detection: 65% detected early, linked to 30% lower mortality in stacked bar plots.

#### 2.2 Bivariate/Multivariate Analysis

* Key Correlations:
Age vs. Age Distribution Score: Strong correlation (ρ=0.86, p<0.001).
Tumor Size vs. Treatment: Surgery preferred for larger tumors (>50mm).
Healthcare Costs: No correlation with outcomes (ρ≈0).

* Mortality Drivers:
Smoking: 58% of smokers died vs. 42% non-smokers.
Physical Activity: Low activity → 2.1× higher mortality vs. high activity.
Genetic Mutation: Present in 35% of deceased patients vs. 22% survivors.

* Visualizations:
Heatmap: Confirmed weak correlations between numerical variables (all |ρ|<0.3).
Pairplot: Showed non-linear relationships (e.g., incidence vs. mortality rates).

### Continent-based Analysis

#### 3.1 Geographic Trends

* ASR Scores: Europe dominated (42.6% of total ASR), reflecting higher CRC burden.
Healthcare Costs:
Asia: Highest median costs ($78K), likely due to advanced care in Japan/South Korea.
Africa: Lowest costs ($45K), but sparse data (only 5% of samples).

#### 3.2 Risk Factors by Continent

* Europe/North America:
Highest rates of obesity (32%) and high-risk diets (processed meat/low fiber).
60% of genetic mutations reported in these regions (likely due to better testing access).

* Asia:
Leading in smoking (45%) and alcohol consumption (52%).
Early Detection: 70% (vs. 55% in Africa).

#### Pie Charts:

* Tumor Size: Asia contributed 35.2% of cases, Europe 26.3%.
Age Distribution: Asia (33.4%) and Europe (26.8%) had the oldest populations.

### Hypothesis Testing

#### 4.1 Objective
Test if there is a significant difference in survival rates between American continents (North/South America) and other continents.

#### 4.2 Methodology
Test: Chi-Square Test of Independence (α=0.05).

* Variables:
Dependent: Survival_5_years (binary: Yes/No).
Independent: Continent (6 categories).

#### 4.3 Results

For Survival_5_years, Chi-Square Statistic is 8.27	with p-value being 0.142, thus failing to reject H₀.
Also for Mortality, Chi-Square Statistic is 6.99 with p-value being 0.222, hence again	failing to reject H₀.

Interpretation:

* Fail to Reject the Null Hypothesis for Survival_5_years_numerical: No significant relationship between continent and survival_5_years_numerical.

* Fail to Reject the Null Hypothesis for Mortality: No significant relationship between continent and mortality.

* These results show that the data does not provide enough information to prove our hypothesis (alternative hypothesis), so we stick with null hypothesis.

* Possible Reasons:
Similar healthcare standards across continents in the dataset.
Data limitations (e.g., underrepresentation of African countries).

#### Key Takeaways

* Age and Lifestyle are stronger predictors of CRC outcomes than geography.

* Early Detection reduces mortality by 30%, which is critical for public health strategies.

* Not enough prove to support Alternative Hypothesis, so we stick to Null Hypothesis: Survival rates are statistically similar across continents.

## Stage III - Machine Learning for Colorectal Cancer Survival Prediction: A Comprehensive Analysis

## 1. Introduction

Colorectal cancer (CRC) is a leading cause of cancer-related deaths worldwide. Early and accurate prediction of survival outcomes can significantly improve treatment planning and patient care. This project applies machine learning (ML) classification techniques to predict 5-year survival in CRC patients using a diverse dataset encompassing clinical, demographic, and lifestyle factors.

### 1.1 Project Objectives

- Develop a binary classification model (`survived` vs. `deceased within 5 years`).
- Evaluate three dataset variants to assess the impact of feature engineering.
- Compare four ML algorithms:
  - Logistic Regression
  - Naïve Bayes
  - Random Forest
  - Gradient Boosting
- Optimize models using hyperparameter tuning and cross-validation.
- Provide interpretable insights for clinical decision-making.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Dataset Overview

The dataset includes:

- **Numerical features** (e.g., age, tumor size).
- **Binary categorical features** (e.g., smoking status, diabetes).
- **Nominal categorical features** (e.g., country, treatment type).

### 2.2 Data Transformation Strategies

Three dataset variants were created for experimentation:

#### 2.2.1 Original Data
- **Binning:** Continuous variables discretized into ordinal bins (e.g., tumor size → small/medium/large).
- **Normalization:** `StandardScaler` applied to numerical features.

#### 2.2.2 Combined-Feature Dataset (`Major_feature_df`)
Six composite features were engineered to capture holistic risk dimensions:

| Composite Feature          | Description |
|---------------------------|-------------|
| `HEALTH_RISK_out_of_9`    | Lifestyle factors (smoking, obesity, diabetes) |
| `DISEASE_RISK_out_of_9`   | Clinical severity (cancer stage, tumor size) |
| `Healthcare_Quality_Index`| Country-level healthcare performance |
| `ECONOMIC_RISK_out_of_5`  | Socioeconomic status, insurance access |
| `Hospital_Treatment_Score`| Treatment intensity, early detection |
| `COUNTRY_BASED_RISK`      | Epidemiological country-level risk trends |

- All features binned into ordinal categories (Low / Moderate / High Risk).

#### 2.2.3 Feature Importance-Based Dataset
- Top predictive features selected from Random Forest and Gradient Boosting models using feature importance scores.

### 2.3 Handling Missing Data & Dimensionality Reduction

- **Missing Values:** Imputed using:
  - Mode for categorical features.
  - Median for numerical features.
- **Dimensionality Reduction:**
  - Applied **PCA** (Principal Component Analysis) to retain 95% variance and reduce dimensionality.

---

## 3. Data Preparation for ML

### 3.1 Motivation for Binning Numerical Features

- **Enhanced Interpretability:** Categories like "High Risk" are more intuitive than raw scores.
- **Model Performance:** Tree-based models benefit from ordinal splits (e.g., "Moderate vs. High Risk").
- **Outlier Robustness:** Reduces sensitivity to extreme values.

### 3.2 Post-Transformation Cleanup

- Dropped original numerical columns (e.g., `HEALTH_RISK_out_of_9`) to:
  - Avoid multicollinearity.
  - Simplify modeling with categorical-only features.
  - Ensure consistency across models.

**Final Dataset:** 6 categorical composite features + target (`Survival_5_years`)

### 3.3 Pipeline Setup

- **Feature/Target Separation:**
  - Removed non-predictive fields (e.g., `Patient_ID`)
  - Target: `Survival_5_years` (encoded as 0/1)

- **Train-Test Split:**
  - 80/20 stratified split to preserve class balance.

- **Feature Processing:**
  - Numerical: Imputed and standardized.
  - Categorical:
    - One-hot encoded (nominal)
    - Label encoded (binary)
    - Target encoded (high-cardinality)

- **Integration:**
  - Used `ColumnTransformer` for preprocessing.
  - Wrapped all transformations using `scikit-learn.Pipeline` to prevent data leakage.

---

## 4. Machine Learning Models & Evaluation

### 4.1 Algorithms Tested

- **Logistic Regression:** Baseline interpretability.
- **Naïve Bayes:** Probabilistic benchmark.
- **Random Forest:** Ensemble model for non-linear interactions.
- **Gradient Boosting (XGBoost / LightGBM):** Optimized sequential learning.

### 4.2 Hyperparameter Tuning

- Used `GridSearchCV` and `RandomizedSearchCV` with 5-fold cross-validation.
- Optimization metric: **ROC-AUC**

#### Key Parameters Tuned:
- **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`
- **Gradient Boosting:** `learning_rate`, `n_estimators`, `subsample`
- **Logistic Regression:** `C`, `penalty` (L1/L2)

### 4.3 Evaluation Metrics

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- F1-Score  
- Specificity  
- ROC-AUC  
- **PR-AUC** (better for imbalanced datasets)  
- Confusion Matrix

---

## 5. Results & Comparative Analysis

### 5.1 Performance Across Dataset Variants

| Model              | Original Data | Major_feature_df | Feature-Selected Data |
|-------------------|---------------|------------------|------------------------|
| Logistic Regression | 0.78          | 0.82             | 0.80                   |
| Naïve Bayes         | 0.75          | 0.79             | 0.77                   |
| Random Forest       | 0.83          | 0.87             | 0.85                   |
| Gradient Boosting   | 0.85          | **0.89**         | 0.86                   |

### 5.2 Key Findings

- ✅ **Best-performing model:** Gradient Boosting on `Major_feature_df` (89% accuracy).
- ✅ **Composite Features Boosted Accuracy:**
  - Aggregating lifestyle + clinical + treatment data improved model performance.
  - `DISEASE_RISK_out_of_9` was the most impactful feature.
- ✅ **Categorical Binning Improved Interpretability:**
  - Clinicians can easily interpret labels like “High Risk” vs. raw scores.
- ✅ **PR-AUC Useful for Imbalanced Classes:**
  - Precision-Recall curve gave better insight into model sensitivity.
- ⚠️ **Potential Data Leakage Risk:**
  - If `COUNTRY_BASED_RISK` includes mortality data, results may be inflated.

---

## 6. Conclusion & Clinical Implications

- ✅ **Top Model:** Gradient Boosting with engineered features from `Major_feature_df`.
- ✅ **Highest Accuracy:** 89%
- ✅ **Interpretable Outputs:** Composite risk scores can aid clinician understanding.
- ✅ **Sensitivity-Prioritized:** Reduces false negatives, improving critical case detection.

### 6.1 Future Work

- External validation using independent datasets.
- Deployment as a clinical decision-support system.
- Incorporate deep learning models to explore more complex feature interactions.

---

## 7. Key Takeaways

- ✅ **Best Practice:** Pipelines + cross-validation prevent data leakage.
- ✅ **Top Model:** Gradient Boosting on categorical composite features.
- ✅ **Clinical Relevance:** High sensitivity ensures critical survival cases are identified.
- ✅ **Scalability:** Modular preprocessing allows for easy dataset extensions.

---
