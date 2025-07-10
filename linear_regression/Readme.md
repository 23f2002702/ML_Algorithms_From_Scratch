# Linear Regression From Scratch

This project demonstrates how to implement **Linear Regression** **from scratch** using only **Python** and **NumPy** — no pre-built `scikit-learn` models for the training step.

**Goal:**  
Predict California house prices based on **Median Income** using a simple linear line:  
\[
\hat{y} = wX + b
\]

---

## Contents

- `01_linear_regression_from_scratch.ipynb` — Fully documented Jupyter Notebook with:
  - Data loading
  - Maths explanation
  - Gradient Descent implementation
  - Visualizations
  - Comparison with `scikit-learn`
- `data/` — Contains dataset CSV (optional, since dataset is auto-fetched)
- `README.md` — This file

---

## Dataset

- **California Housing Dataset**
- Source: `sklearn.datasets.fetch_california_housing`
- Predicts median house value based on features like:
  - Median Income (used here)
  - House Age
  - Average Rooms
  - Population, etc.

---

## ⚙️ How It Works

### Hypothesis

Predict house price with:
\[
\hat{y} = wX + b
\]

---

### Loss Function

Minimize the **Mean Squared Error (MSE)**:
\[
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

---

### Optimization

- Uses **Gradient Descent** to update `w` and `b` step by step.
- Adjusts parameters to reduce prediction error.

---

### Visualizations

- Scatter plot: Median Income vs Price
- Best-fit line: shows how well the model fits
- Loss curve: shows how error decreases over epochs
- Comparison with `scikit-learn`’s `LinearRegression` for verification

---

### Results
- Trained a simple Linear Regression model without sklearn.
- Compared its slope (w) and intercept (b) with scikit-learn to verify correctness.
- Visualized how Gradient Descent works in practice.

---

### Author
Shilajit Mukherjee

