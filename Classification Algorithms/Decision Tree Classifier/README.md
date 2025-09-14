# Decision Tree Classifier From Scratch  

This project demonstrates how to implement a **Decision Tree Classifier** **from scratch** using only **Python** and **NumPy** — no pre-built `scikit-learn` models for the training step (only for benchmarking).  

**Goal:**  
Classify samples using a **tree-based model** that recursively splits the data based on **impurity measures** (Gini Index / Entropy).  

---

# Contents  

- `Decision_Tree_Classifier_From_Scratch.ipynb` — Fully documented Jupyter Notebook with:  
  - Data loading (Bank Marketing dataset / Iris dataset for demo)  
  - Impurity measures: **Gini Index**, **Entropy**  
  - Recursive tree building from scratch  
  - Prediction function (tree traversal)  
  - Comparison with `scikit-learn`’s `DecisionTreeClassifier`  
  - Accuracy, F1-score, baseline comparisons  
  - Regularization (max_depth, min_samples_split)  
  - Advanced visualizations (tree structure, decision boundaries, PR curve, learning curve)  

- `data/` — Contains `train.csv` and `test.csv` for the Bank Marketing dataset  
- `README.md` — This file  

---

# Dataset  

### Iris Dataset (Demo)  (To run a demo, dataset not provided here)
- Source: `sklearn.datasets.load_iris`  
- Target: Multiclass classification → Setosa, Versicolor, Virginica  

### Bank Marketing Dataset (Full)  
- Source: UCI Machine Learning Repository  
- Features: **16 attributes** (age, job, education, balance, etc.)  
- Target: Binary class → Will the client **subscribe to a term deposit?** (Yes/No)  
- Data split: `train.csv` (~45k rows), `test.csv` (~4.5k rows)  

---

# ⚙️ How It Works  

## Impurity Measures  

- **Gini Index**:  

$$
Gini(S) = 1 - \sum_{i=1}^{c} p_i^2
$$  

- **Entropy (Information Gain)**:  

$$
Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i)
$$  

Splits are chosen to **maximize Information Gain** or **minimize Gini impurity**.  

---

## Tree Building  

- Recursively choose the **best feature and threshold** to split the dataset.  
- Stop when a **maximum depth**, **minimum samples per split**, or **pure leaf** is reached.  
- Leaves store the majority class of that node.  

---

## Prediction  

- Traverse the tree from root → leaf following feature thresholds.  
- The leaf node’s class becomes the prediction.  

---

## Regularization  

To prevent **overfitting**, the implementation supports:  
- `max_depth` — Limit tree depth  
- `min_samples_split` — Minimum samples required to split further  

---

# Visualizations  

- **Decision Boundaries** (Train vs Test, Scratch vs Sklearn)  
- **Tree Structure** (sklearn plot)  
- **Class Distribution** (target imbalance)  
- **Learning Curve** (train vs test accuracy)  
- **Precision-Recall Curve** (imbalanced dataset performance)  

---

# Results  

| Model                     | Accuracy (Test) | F1 Score | Notes                          |
|----------------------------|-----------------|----------|--------------------------------|
| Scratch Decision Tree      | ~0.88–0.93      | ~0.90    | Works well, slight overfitting |
| Sklearn DecisionTree       | ~0.90–0.95      | ~0.91    | Similar performance, faster    |
| Baseline (Majority Class)  | ~0.88           | 0.00     | Predicts "No" for all cases    |

- Scratch implementation is **comparable to sklearn**.  
- Regularization (`max_depth`, `min_samples_split`) helps reduce overfitting.  
- Far better than baseline (majority class).  

---

# Author  

**Shilajit Mukherjee**  
- Data Science student at IITM  
- AI/ML Enthusiast  

For queries, feel free to reach out!  
