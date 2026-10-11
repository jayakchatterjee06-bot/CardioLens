# Model evaluation

Repeated stratified 5-fold cross-validation (5 repeats, 25 fits per model). Values are mean ± std across folds. Brier score: lower is better. ★ = selected (highest ROC-AUC).

## Selected models

| Target | Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier |
|---|---|---|---|---|---|---|---|
| Overall CAD | Random Forest | 0.923 ± 0.037 | 0.857 ± 0.043 | 0.864 ± 0.033 | 0.950 ± 0.036 | 0.904 ± 0.029 | 0.115 ± 0.014 |
| Left Anterior Descending | Random Forest | 0.852 ± 0.047 | 0.786 ± 0.048 | 0.777 ± 0.047 | 0.893 ± 0.059 | 0.829 ± 0.039 | 0.166 ± 0.014 |
| Left Circumflex | Random Forest | 0.729 ± 0.065 | 0.676 ± 0.056 | 0.657 ± 0.137 | 0.361 ± 0.102 | 0.462 ± 0.113 | 0.207 ± 0.014 |
| Right Coronary Artery | Logistic Regression | 0.728 ± 0.055 | 0.669 ± 0.048 | 0.572 ± 0.074 | 0.481 ± 0.108 | 0.518 ± 0.088 | 0.209 ± 0.025 |

## Final deployed models (selected model + Platt calibration)

Out-of-fold predictions; the calibrator for each patient was fitted without that patient. Threshold 0.5.

| Target | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier (raw → calibrated) |
|---|---|---|---|---|---|---|
| Overall CAD | 0.915 | 0.871 | 0.893 | 0.931 | 0.912 | 0.116 → 0.100 |
| Left Anterior Descending | 0.851 | 0.789 | 0.796 | 0.859 | 0.826 | 0.164 → 0.152 |
| Left Circumflex | 0.723 | 0.696 | 0.652 | 0.487 | 0.558 | 0.207 → 0.206 |
| Right Coronary Artery | 0.712 | 0.657 | 0.566 | 0.377 | 0.453 | 0.214 → 0.205 |

## Overall CAD (`Cath`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.853 ± 0.047 | 0.891 ± 0.041 | 0.907 ± 0.047 | 0.898 ± 0.033 | 0.923 ± 0.032 | 0.101 ± 0.025 |
| Random Forest ★ | 0.857 ± 0.043 | 0.864 ± 0.033 | 0.950 ± 0.036 | 0.904 ± 0.029 | 0.923 ± 0.037 | 0.115 ± 0.014 |
| Gradient Boosting | 0.855 ± 0.044 | 0.885 ± 0.038 | 0.918 ± 0.036 | 0.901 ± 0.030 | 0.900 ± 0.045 | 0.108 ± 0.030 |

Top features by mean |SHAP|: Typical chest pain (0.113), Atypical chest pain (0.054), Age (0.052), Ejection fraction (TTE) (0.036), Regions with wall-motion abnormality (0.035)

## Left Anterior Descending (`LAD`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.744 ± 0.046 | 0.782 ± 0.051 | 0.784 ± 0.070 | 0.781 ± 0.042 | 0.830 ± 0.051 | 0.171 ± 0.029 |
| Random Forest ★ | 0.786 ± 0.048 | 0.777 ± 0.047 | 0.893 ± 0.059 | 0.829 ± 0.039 | 0.852 ± 0.047 | 0.166 ± 0.014 |
| Gradient Boosting | 0.747 ± 0.038 | 0.774 ± 0.047 | 0.806 ± 0.052 | 0.788 ± 0.030 | 0.812 ± 0.045 | 0.179 ± 0.026 |

Top features by mean |SHAP|: Typical chest pain (0.111), Ejection fraction (TTE) (0.064), Regions with wall-motion abnormality (0.060), Age (0.055), Atypical chest pain (0.047)

## Left Circumflex (`LCX`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.653 ± 0.070 | 0.571 ± 0.124 | 0.476 ± 0.104 | 0.517 ± 0.107 | 0.692 ± 0.076 | 0.227 ± 0.035 |
| Random Forest ★ | 0.676 ± 0.056 | 0.657 ± 0.137 | 0.361 ± 0.102 | 0.462 ± 0.113 | 0.729 ± 0.065 | 0.207 ± 0.014 |
| Gradient Boosting | 0.677 ± 0.052 | 0.608 ± 0.091 | 0.495 ± 0.106 | 0.542 ± 0.095 | 0.722 ± 0.066 | 0.215 ± 0.030 |

Top features by mean |SHAP|: Age (0.072), Typical chest pain (0.053), Platelets (0.035), Creatinine (0.035), Triglycerides (0.033)

## Right Coronary Artery (`RCA`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Logistic Regression ★ | 0.669 ± 0.048 | 0.572 ± 0.074 | 0.481 ± 0.108 | 0.518 ± 0.088 | 0.728 ± 0.055 | 0.209 ± 0.025 |
| Random Forest | 0.672 ± 0.037 | 0.633 ± 0.116 | 0.319 ± 0.095 | 0.416 ± 0.095 | 0.716 ± 0.055 | 0.206 ± 0.010 |
| Gradient Boosting | 0.679 ± 0.047 | 0.593 ± 0.078 | 0.491 ± 0.121 | 0.529 ± 0.088 | 0.693 ± 0.066 | 0.224 ± 0.030 |

Top features by mean |SHAP|: Diabetes mellitus (0.055), Age (0.049), Sex (male) (0.042), Functional class (0.038), Dyspnea (0.036)
