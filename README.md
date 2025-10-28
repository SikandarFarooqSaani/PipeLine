# 🤖 This README is AI Generated (Custom Prompt) 😊  
*For reader clarity and my own ease of documentation.*

---

# 🧩 Understanding Pipelines in Machine Learning

## 📦 Dataset
- **Dataset:** Famous **Titanic dataset** (available in repo).  
- **Goal:** To learn how to build, use, and export a machine learning pipeline that automates preprocessing and model training.

---

## 🎯 Objective
Pipelines are used to **sequentially organize transformations and modeling steps**.  
They make the workflow clean, reproducible, and ready for **deployment**.  
Instead of running each transformation manually, all are executed **in order** through one unified structure.

---

## ⚙️ Concept Overview
A **Pipeline** in scikit-learn allows chaining multiple data preprocessing steps and a model into one object.  
Once defined, you can:
- Fit the pipeline with data.
- Transform and train automatically.
- Predict and evaluate performance.
- Export it directly for deployment.

---

## 🧠 Steps Performed

### 1️⃣ Importing Required Tools
- pandas, numpy  
- sklearn:
  - `Pipeline`, `make_pipeline`
  - `ColumnTransformer`
  - `SimpleImputer`, `OneHotEncoder`
  - `MinMaxScaler`, `SelectKBest`, `chi2`
  - `DecisionTreeClassifier`

---

### 2️⃣ Defining Transformations
Each step of the preprocessing pipeline was defined separately:

#### 🔹 Transformer 1: Imputation
Handles missing values:
```python
trf1 = ColumnTransformer([
    ('age_imputer', SimpleImputer(strategy='mean'), ['Age']),
    ('embark_imputer', SimpleImputer(strategy='most_frequent'), ['Embarked'])
])
🔹 Transformer 2: Encoding

Encodes categorical features using OneHotEncoder:

trf2 = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=False), ['Sex', 'Embarked'])
])

🔹 Transformer 3: Scaling

Scales numerical values using MinMaxScaler:

trf3 = MinMaxScaler()

🔹 Transformer 4: Feature Selection

Selects top 8 important features using Chi-Square test:

trf4 = SelectKBest(score_func=chi2, k=8)

🔹 Transformer 5: Model

Defines the model to be trained:

trf5 = DecisionTreeClassifier()

3️⃣ Building the Pipeline

All transformations and model are combined in one pipeline:

from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('imputer', trf1),
    ('encoder', trf2),
    ('scaler', trf3),
    ('selector', trf4),
    ('model', trf5)
])


You can also use:

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(trf1, trf2, trf3, trf4, trf5)


make_pipeline is simpler — it doesn’t require naming each step manually.

4️⃣ Fitting and Evaluating

Fit the pipeline:

pipe.fit(X_train, y_train)


Predict and evaluate accuracy:

y_pred = pipe.predict(X_test)
accuracy_score(y_test, y_pred)


Check transformation details:
You can inspect any stage using indexing, e.g.:

pipe.named_steps['scaler']

5️⃣ Cross Validation and Optimization

Pipelines integrate smoothly with model tuning tools:

from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

cross_val_score(pipe, X, y, cv=5)
GridSearchCV(pipe, param_grid, cv=5)
RandomizedSearchCV(pipe, param_distributions, cv=5)

6️⃣ Deployment

Once satisfied with performance, the entire pipeline can be exported using joblib or pickle:

import joblib
joblib.dump(pipe, 'titanic_pipeline.pkl')


This saves all transformations and the model — ready for direct use in production.

### 🧾 Summary

✅ Pipelines help automate complex workflows.
✅ They maintain data consistency from training to deployment.
✅ Each preprocessing and model step is organized cleanly.
✅ Easily compatible with hyperparameter tuning and evaluation tools.

📁 Outcome:
A complete, reusable, and deployable pipeline trained on the Titanic dataset — transforming, encoding, scaling, selecting features, and classifying in one streamlined process.



<img width="365" height="341" alt="pipe1" src="https://github.com/user-attachments/assets/3c47ae63-1978-4a7e-917a-93338ecdd288" />


<img width="389" height="367" alt="pipe2" src="https://github.com/user-attachments/assets/db214316-e4d3-434b-9d9c-b6d45a0748cc" />

