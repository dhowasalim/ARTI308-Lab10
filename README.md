# ARTI308 – Support Vector Machines

Classification with **Support Vector Machines (SVC)** and hyperparameter tuning with **GridSearchCV** in `scikit-learn`.

| File | Description |
|------|-------------|
| `01-Support_Vector_Machines.ipynb` | Lab: breast cancer classification (scikit-learn built-in dataset) |
| `02-SVM_Assignment.ipynb` | Assignment: classifying the 3 Iris flower species |

No CSV files are needed: the lab uses `sklearn.datasets.load_breast_cancer()` and the assignment uses `seaborn.load_dataset('iris')` (needs an internet connection the first time).

## Assignment – Iris
EDA (pairplot, KDE of sepal length vs width for setosa) → 70/30 split → `SVC()` → GridSearchCV over `C ∈ {0.1, 1, 10, 100}` and `gamma ∈ {1, 0.1, 0.01, 0.001}`.

| Model | Accuracy |
|---|---|
| Default SVC | 0.98 |
| Grid search (C=10, gamma=0.01) | 0.98 |

Setosa is perfectly separable; the only error is one versicolor flower predicted as virginica.

## Run it
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook
```
