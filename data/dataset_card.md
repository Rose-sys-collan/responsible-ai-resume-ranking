# Dataset Card

## 1. Source
Source: https://www.kaggle.com/datasets/ramzybakir/ai-generated-resume-dataset/data

Attempted to clean data from the above dataset, discarded as the info was skewed and lacked info
After two 2 days of searching, I decided to choose SYRA Generated dataset, although generated, it had info I needed (names, nationality, etc)

---  

## 2. Schema
| Column | Type | Description  | Used for training? |
|---------|------|------------- | -------- |
| candidate id | int | used as primary keys since candidate name might be duplicated | ✅ |
| name | string | candidate name | 🤔 |
| job_title | string | the name of the position that candidate applies for, part 1 of JD data | ✅ |
| job_description | string | full job description text, part 2 of JD data | ✅ |
| skills | string | comma-separated list of skills | ✅ |
| education | string | highest degree, field of study, certificate | ✅ |
| experience | string | short summary of previous work experience | ✅ |
| gender | categorical | female / male / other / unknown | ❌ |
| age | categorical | e.g., "20–29", "30–39" | ❌ |
| nationality | categorical | standardized country code (CA, CN, IN, etc.) | ❌ |

---  

## 3. Protected Attributes  
The following columns are considered **protected attributes** — sensitive information used **only for fairness evaluation**, not for model training or ranking.

| Attribute | Description | Notes |
|------------|-------------|-------|
| gender | candidate gender | extracted or inferred; may contain uncertainty |
| age | approximate age group | derived from resume text or metadata |
| nationality | country code (CA, CN, IN, etc.) | derived from location field |

**Usage Policy**  
(i) These fields must **not** be used as input features for similarity, ranking, or embedding models.  
(ii) They are included solely for fairness evaluation metrics such as demographic parity and equal opportunity.  
(iii) Any files shared publicly or used for visualization should mask or anonymize these attributes.

---  

## ⚙️ 5. Preprocessing Steps

| Step | Description |  
|------|--------------|  
| 1 | Loaded raw data from `data/raw/` |  
| 2 | Removed duplicates and incomplete records |  
| 3 | Standardized text (lowercase, punctuation removal, etc.) |  
| 4 | Extracted and labeled protected attributes (`gender`, `age`, `nationality`) |  
| 5 | Saved cleaned output to `data/cleaned/cleaned_dataset.csv` |  
