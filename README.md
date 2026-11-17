<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/Kaggle-Dataset-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle" />
  <img src="https://img.shields.io/badge/Best%20Accuracy-91.23%25-success" alt="91.23% accuracy" />
</p>

# Harvest Vision

**A crop-yield classification pipeline** benchmarking six scikit-learn models on a 100,000-row sample of a 1,000,000-row agricultural dataset. Yield in tons per hectare is converted to a binary High/Low target by median split, then six classifiers are trained, tuned and compared with confusion matrices and learning curves.

> **Domain:** Precision agriculture and crop-yield prediction
> **Task:** Binary classification (High / Low yield) via median split on `Yield_tons_per_hectare`
> **Notebook:** [`yield-prediction-1-0.ipynb`](yield-prediction-1-0.ipynb) &middot; [run on Kaggle](https://www.kaggle.com/code/mohamedaymanelkhatib/harvest-vision)

---

## Results

Every model is scored on the same held-out 19,998-row test split.

| # | Model | Accuracy |
|---|-------|----------|
| 1 | **Stacking Classifier** (RF + Linear SVM to Logistic Regression) | **0.912341** |
| 2 | Linear SVM (`LinearSVC`) | 0.911791 |
| 3 | Logistic Regression | 0.911341 |
| 4 | Random Forest (100 trees) | 0.908041 |
| 5 | K-Nearest Neighbours (tuned, k=7 uniform) | 0.877888 |
| 6 | Decision Tree (`max_depth=3`) | 0.849285 |

**Best model: the Stacking Classifier at 91.23% accuracy.**

The honest reading of that table: stacking wins, but by 0.05 percentage points over a plain Linear SVM. The top three models sit inside a 0.1-point band, which says the decision boundary here is close to linear and the ensemble is not buying much. The depth-3 Decision Tree is 6.3 points behind, and that gap is the one that reflects real model capacity rather than noise.

Per-class precision, recall and F1 are all 0.91 for the stacking model, and the test split is almost perfectly balanced (9,976 vs 10,022), so accuracy is a fair headline metric here.

### KNN tuning

`GridSearchCV` (3-fold, `f1_weighted`) over `n_neighbors` in {3, 5, 7} and `weights` in {uniform, distance}:

| | Accuracy |
|---|---|
| Untuned (k=5) | 0.876588 |
| **Tuned (k=7, uniform)** | **0.877888** |

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Agriculture Crop Yield (Kaggle)](https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield) |
| Total rows | 1,000,000 |
| Working sample | 100,000 (`random_state=42`) |
| After IQR outlier removal | 99,988 |
| Train / test | 79,990 / 19,998 (80/20, `random_state=42`) |
| Features | 9 (4 numerical, 5 categorical) |
| Target | `Yield_tons_per_hectare` to binary High/Low on the median |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `Region` | Categorical | Geographical location |
| `Soil_Type` | Categorical | Sandy, clay, loamy, and so on |
| `Crop` | Categorical | Wheat, corn, rice, and so on |
| `Weather_Condition` | Categorical | Weather over the growing period |
| `Irrigation_Used` | Boolean | Whether irrigation was applied |
| `Fertilizer_Used` | Boolean | Whether fertilizer was applied |
| `Rainfall_mm` | Numerical | Total rainfall in millimetres |
| `Temperature_Celsius` | Numerical | Average temperature |
| `Days_to_Harvest` | Numerical | Days from planting to harvest |

The CSV is about 89 MB and is **not** committed here. Download it from the Kaggle link above, or run the notebook on Kaggle where it is already attached.

---

## Method

1. **EDA** — histograms with KDE and box plots across the four numerical features.
2. **Outlier removal** — IQR filtering (1.5 x IQR) on `Rainfall_mm`, `Temperature_Celsius`, `Days_to_Harvest` and `Yield_tons_per_hectare`. Removes 12 rows of 100,000.
3. **Encoding** — `LabelEncoder` on the four categorical columns; booleans cast to int.
4. **Target** — binary class from a median split on `Yield_tons_per_hectare`.
5. **Split and scale** — 80/20 split, then `StandardScaler` fitted on the training partition only and applied unchanged to the test partition, so no test statistics leak into preprocessing.
6. **Class balancing** — SMOTE (`random_state=42`) on the training set only.
7. **Train and evaluate** — six models, each with a classification report and a confusion matrix.
8. **Learning curves** — 3-fold CV across five training sizes for all six models, to separate underfitting from overfitting.

### Model configuration

| Model | Configuration |
|-------|---------------|
| Decision Tree | `max_depth=3` (kept shallow so the tree stays readable) |
| Logistic Regression | `max_iter=1000`, L2 |
| Linear SVM | `LinearSVC`, `max_iter=1000` |
| K-Nearest Neighbours | `ball_tree`, tuned by `GridSearchCV` on a 10% subsample of the balanced training set |
| Random Forest | 100 estimators, `n_jobs=-1` |
| Stacking | Random Forest + Linear SVM, Logistic Regression meta-learner |

---

## Reproducing

Every step is seeded with `random_state=42`, so a clean run reproduces the table above exactly.

```bash
git clone https://github.com/Skapos-crypto/HarvestVision.git
cd HarvestVision
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn graphviz pydotplus
jupyter notebook yield-prediction-1-0.ipynb
```

Point the `read_csv` call in the data-loading cell at your local copy of `crop_yield.csv`, then run all cells. On Kaggle the path is already correct and the dataset is attached.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Language | Python 3.10 |
| ML | scikit-learn (Decision Tree, Logistic Regression, LinearSVC, KNN, Random Forest, Stacking) |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Class balancing | imbalanced-learn (SMOTE) |
| Tuning | GridSearchCV (3-fold) |
| Environment | Jupyter, Kaggle Kernels |

---

## Takeaways

- On this dataset the problem is close to linearly separable: Logistic Regression and a Linear SVM both land within 0.1 points of the stacking ensemble.
- Stacking still gives the top score, but at a real cost in training time for a 0.05-point gain over its own Linear SVM base estimator.
- A depth-3 Decision Tree is the only model meaningfully behind, which is a constraint of the chosen depth rather than of trees generally.
- KNN gains little from tuning here (+0.13 points), and `uniform` weighting beat `distance`.
- SMOTE matters less than expected because the median split produces near-balanced classes by construction.
