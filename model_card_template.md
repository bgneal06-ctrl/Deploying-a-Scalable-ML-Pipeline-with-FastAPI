# 🧾 Model Card

## Model Details
This model is a **binary classification model** trained to predict whether an individual’s annual income exceeds \$50,000 based on U.S. Census data.  

- **Algorithm:** Random Forest Classifier (from `scikit-learn`)  
- **Pipeline:** Preprocessing (OneHotEncoder for categorical features, LabelBinarizer for labels) → Model Training → Inference  
- **Source:** Udacity Machine Learning DevOps Engineer Nanodegree project — *Deploying a Scalable ML Pipeline with FastAPI*  
- **Author:** Brandi Neal  
- **Version:** 1.0  
- **Date:** October 2025  

---

## Intended Use
The model is intended **for educational purposes** and to demonstrate best practices in MLOps, including training, deployment, and performance monitoring using FastAPI.  
It is **not intended for real-world decision-making** such as hiring, credit scoring, or policy decisions.

**Intended Users:**  
- Students and practitioners learning MLOps and model deployment  
- Reviewers assessing pipeline design and monitoring  

---

## Training Data
- **Dataset:** U.S. Census Adult Income dataset (publicly available from the UCI Machine Learning Repository)  
- **Target Variable:** `salary` (`<=50K` vs `>50K`)  
- **Number of Records:** ~32,000 samples  
- **Features:**
  - **Categorical:** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
  - **Continuous:** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`  

---

## Evaluation Data
- **Split:** 80% training, 20% test (randomized using `train_test_split`)  
- **Processing:** Both train and test sets were processed using the same one-hot encoder and label binarizer trained on the training data.  

---

## Metrics
The model was evaluated using **precision**, **recall**, and **F1-score**.

### Overall Test Performance

| Metric | Score |
|:-------|:------|
| Precision | ~0.73 |
| Recall | ~0.66 |
| F1-score | ~0.69 |

---

### Slice-based Performance Highlights
Performance varies across demographic and categorical slices.

#### By Workclass
| Category | F1-score |
|:----------|:----------|
| Federal-gov / Local-gov / Self-employed | 0.72–0.79 |
| Private / State-gov | 0.68–0.72 |
| Without-pay | Unreliable (small sample) |

#### By Education
| Category | F1-score |
|:----------|:----------|
| Doctorate / Masters / Prof-school | 0.84–0.88 |
| HS-grad / 10th–11th grade | 0.33–0.51 |

#### By Sex
| Category | F1-score |
|:----------|:----------|
| Female | 0.60 |
| Male | 0.70 |

#### By Race
| Category | F1-score |
|:----------|:----------|
| White / Black | 0.67–0.68 |
| Amer-Indian-Eskimo | Variable (small sample) |

These results suggest the model performs **reasonably well overall**, but there are **disparities across slices**.

---

## Ethical Considerations
- The dataset reflects **real-world socioeconomic biases**, including historical inequalities in income by gender, race, and education.  
- Predictions may perpetuate these biases if used in decision-making contexts.  
- The model should **not be deployed for any purpose affecting individuals’ opportunities** (e.g., employment, housing, or credit).  
- Monitoring fairness metrics and conducting bias audits are recommended before real-world use.  

---

## Caveats and Recommendations
- **Data Imbalance:** The dataset is skewed toward lower-income classes.  
- **Slice Variance:** Some small slices (e.g., rare countries or occupations) produce unreliable metrics.  
- **Generalization:** The model’s performance may degrade on newer census data or data from other regions.  
- **Recommendation:** Retrain and evaluate periodically; implement fairness checks and consider resampling or class weighting to mitigate imbalance.  

---
