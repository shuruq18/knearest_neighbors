# K-Nearest Neighbors — Telecom Customer Classification

A supervised machine learning project applying the K-Nearest Neighbors (KNN) algorithm to classify telecom customers into service categories based on demographic data.

---

## Overview

This project uses the KNN algorithm to predict which service plan a telecom customer belongs to, based on demographic features like age, income, region, and employment. The notebook walks through the full ML pipeline — from data loading and preprocessing to model training, evaluation, and finding the optimal value of K.

---

## What is K-Nearest Neighbors?

KNN is a distance-based classification algorithm. To predict the class of a new data point, it looks at the K closest points in the training set and assigns the majority class among them. The choice of K significantly affects performance — a small K can be noisy while a large K can oversmooth boundaries.

---

## Dataset

- **Source:** IBM Skills Network — `teleCust1000t.csv` (loaded directly from URL, no download needed)
- **Samples:** 1,000 telecom customers
- **Features:** 11 demographic variables — region, tenure, age, marital status, address, income, education, employment, retirement status, gender, and residence type
- **Target (`custcat`):** 4 customer service groups

| Class | Label | Count |
|---|---|---|
| 1 | Basic Service | 266 |
| 2 | E-Service | 217 |
| 3 | Plus Service | 281 |
| 4 | Total Service | 236 |

---

## Project Pipeline

**1. Data Loading & Exploration** — load CSV, inspect class distribution, visualize income histogram

**2. Feature Engineering** — select 11 demographic features, convert to NumPy array

**3. Normalization** — standardize features to zero mean and unit variance using `StandardScaler` (essential for distance-based algorithms like KNN)

**4. Train/Test Split** — 80/20 split (`test_size=0.2`, `random_state=4`) → 800 training / 200 test samples

**5. Model Training** — fit `KNeighborsClassifier` with initial `k=28`

**6. Evaluation** — measure train and test accuracy using `accuracy_score`

**7. K Optimization** — loop through K values 1–9, plot mean accuracy with standard deviation bands, and identify the best K

---

## Project Structure

```
├── knearest_neighbors.ipynb   # Main notebook
```

---

## Requirements

```
scikit-learn
numpy
pandas
matplotlib
```

Install dependencies with:

```bash
pip install scikit-learn numpy pandas matplotlib
```

---

## Usage

1. Clone the repository and navigate to the project folder.
2. Open and run the notebook:

```bash
jupyter notebook knearest_neighbors.ipynb
```

The dataset is fetched directly from a URL — no local files needed.

---

## Key Concepts

- **KNN** — predicts class based on majority vote among K nearest neighbors in feature space
- **StandardScaler** — normalizes features so no single variable dominates the distance calculation
- **K Optimization** — iterating over K values and plotting accuracy helps select the best tradeoff between bias and variance
- **Out-of-sample accuracy** — evaluating on unseen test data gives a more realistic measure of model performance than training accuracy alone

---

## References

- [scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- [IBM Developer Skills Network — ML0101EN](https://www.coursera.org/learn/machine-learning-with-python)# knearest_neighbors
In this Lab you will load a customer dataset, fit the data, and use K-Nearest Neighbors to predict a data point. But what is K-Nearest Neighbors?
