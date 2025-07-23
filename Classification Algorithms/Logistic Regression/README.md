# Logistic Regression From Scratch

This project demonstrates how to implement **Logistic Regression** **from scratch** using only **Python** and **NumPy** — no pre-built `scikit-learn` models for the training step (only for benchmarking).

**Goal:**  
Classify **Iris flowers** as **Setosa** *(1)* or **Not-Setosa** *(0)* using a simple **Sigmoid-based linear model**:

$$
P(y = 1 | X) = \sigma(wX + b) = \frac{1}{1 + e^{-(wX + b)}}
$$

---

# Contents

- `02_logistic_regression_from_scratch.ipynb` — Fully documented Jupyter Notebook with:
  - Data loading (Iris dataset)
  - Binary classification: Setosa vs Not-Setosa
  - Maths explanation: Sigmoid, Cross-Entropy Loss
  - Gradient Descent implementation
  - Decision Boundary visualization
  - Comparison with `scikit-learn`
  - Regularization

- `data/` — (Optional if you save CSV locally, but uses `sklearn.datasets.load_iris` by default)
- `README.md` — This file

---

# Dataset

- **Iris Dataset**
- Source: `sklearn.datasets.load_iris`
- Features used: *Sepal Length* & *Sepal Width* (2D for easy plotting)
- Target: Binary class → Is the flower **Setosa** or **Not-Setosa**

---

# ⚙️ How It Works

## Hypothesis

Predict probability of class 1:

$$
P(y = 1 | X) = \sigma(wX + b)
$$

where $\sigma$ is the **Sigmoid function**.

---

## Loss Function

Minimize the **Cross-Entropy Loss**:

$$
J(w) = -\frac{1}{m} \sum_{i=1}^{m} [ y^{(i)} \log(h^{(i)}) + (1 - y^{(i)}) \log(1 - h^{(i)}) ]
$$

---

## Optimization

- Uses **Gradient Descent** to update weights and bias.
- Optionally includes **L2 Regularization** to prevent overfitting.
- Step by step updates reduce classification error.

---

## Visualizations

- Scatter plot: Iris flowers by Sepal Length & Width
- Decision Boundary: shows how the model splits classes
- Sigmoid Curve: maps linear output to probability
- Cost History: shows how loss decreases over epochs
- Comparison with `scikit-learn`’s `LogisticRegression` for verification

---

## Results

- Trained a **Binary Logistic Regression** classifier without sklearn.
- Compared weights, bias, and decision boundary with `scikit-learn` to verify correctness.
- Added regularization to control overfitting.
- Visualized probability output and boundary for deeper understanding.

---

## Author

Shilajit Mukherjee  
- Data Science student at IITM  
- AI/ML Enthusiast  

For queries, feel free to reach out!
