# Colorectal Cancer Survival Analysis - DSA210 Project

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

### 4. Data Description & Collection
   
The project will use the following datasets:

Colorectal Cancer Dataset:
As briefly mentioned before this dataset contains 26 parameters (e.g., age, gender, stage of cancer, country, survival outcome) across over 160,000 records of colorectal cancer patients. The parameters are a mix of categorical and numerical variables that describe clinical, demographic, and medical factors influencing colorectal cancer prognosis. Since the data is publicly available and open to being overly used, the enrishment of the data from different sources may be required. The data augmentation is planned as following:

Age distribution score: Calculated using data from the United Nations, this score categorizes the global population into age groups, which will be assigned to corresponding patients in the dataset. The estimated distrubution of the new cases of colorectal cancer with respect to age is the following: 150 cases among 0 to 19 years old, 6,880 cases among 20 to 44 years old, 138,722 cases among 45 to 69 years old, and 195,667 cases 70+ years old. These numbers will be standardized and will be added to the data to enrich the data. Age distribution scores (ADS) are listed in corresponding order: 0.00044 (age: 0-19), 0.02015 (age: 20-44), 0.40631(age: 45-69), 0.57310 (Age: 70+).

Death rates by country (ASR): This score, based on country-specific mortality rates, will enhance the model's ability to account for the impact of healthcare systems and country-level factors on colorectal cancer survival. It is planned to used these scores and implement them in my data using reported country information for each student, which I believe will play a significant role in decision-making performance of the ML model. Sources: 

[Mustafa Derin 32272]

Sources:

1- https://encr.eu/sites/default/files/inline-files/Colorectal_cancer_factsheet_March_2021.pdf [ENCR] 

2- https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions

3- Wang, C. C., Sung, W. W., Yan, P. Y., Ko, P. Y., & Tsai, M. C. (2021). Favorable colorectal cancer mortality-to-incidence ratios in countries with high expenditures on health and development index: A study based on GLOBOCAN database. Medicine, 100(41), e27414. https://doi.org/10.1097/MD.0000000000027414

4- https://www.wcrf.org/preventing-cancer/cancer-statistics/colorectal-cancer-statistics/ [WCRF] 
