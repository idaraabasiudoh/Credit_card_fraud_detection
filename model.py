# Uncomment these if required
# pip install scikit-learn==1.0.2
# !pip install snapml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
import time
import warnings
import snapml
from snapml import SupportVectorMachine

warnings.filterwarnings('ignore')


# Dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

raw_data = pd.read_csv(url)
print("There are " + str(len(raw_data)) + " credit card observations from the dataset")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset")

# Inflating original dataset
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, 10, axis=0), columns=raw_data.columns)

print("\nThere are " + str(len(big_raw_data)) + " credit card observations from the dataset")
print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset")

# Get set of distinct classes
labels = big_raw_data.Class.unique()

# Get the count of each class
sizes = big_raw_data.Class.value_counts().values

print(labels)
print(sizes)

# Plotting class column to visualize data
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Target Class Values")
plt.show()


# Dataset preprocessing
# Standardize data
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values

# X: Feature Matrix (Excluding index 1 - Time variable)
# y: Label vectors
X = data_matrix[:, 1:30]
y = data_matrix[:, 30]

# Data Normalization
X = normalize(X, norm="l1")

# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)


# Dataset Train/Test split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('X_train.shape=', X_train.shape, 'y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'y_test.shape=', y_test.shape)


# Building a Decision Tree Classifier with scikit-learn
weight = compute_sample_weight('balanced', y_train)

sk_dt = DecisionTreeClassifier(max_depth=4, random_state=35)
t0 = time.time()
sk_dt.fit(X_train, y_train, sample_weight=weight)
sk_speed = time.time() - t0
print(f"Scikit-Learn training time(s): {sk_speed:.2f}")

# Building a Decision Tree Classifier with SnapML
snap_dt = snapml.DecisionTreeClassifier(max_depth=4, random_state=45)
t0 = time.time()
snap_dt.fit(X_train, y_train, sample_weight=weight)
snap_speed = time.time() - t0
print(f"Snap Ml training time(s): {snap_speed:.2f}")

# Evaluating model speed and accuracy
training_speedup = sk_speed / snap_speed
print(f"[Training time speedup] Snap ML vs Scikit-Learn: {training_speedup}")

sklearn_pred = sk_dt.predict(X_test)
snapml_pred = snap_dt.predict(X_test)
print(f"[Scikit-Learn Confusion Matrix: \n{confusion_matrix(y_test, sklearn_pred)}")
print(f"[Snap ML Confusion Matrix: \n{confusion_matrix(y_test, snapml_pred)}")
print(f"[Scikit-Learn Classification Report: \n{classification_report(y_test, sklearn_pred)}")
print(f"[Snap ML Classification Report: \n{classification_report(y_test, snapml_pred)}")

# Building a Support Vector Machine using Scikit-Learn
sk_svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
t0 = time.time()
sk_svm.fit(X_train, y_train)
sk_speed = time.time() - t0
print(f"Scikit-Learn training time(s): {sk_speed:.2f}")

# Building a Support Vector Machine using Snap ML
snap_svm = SupportVectorMachine(class_weight='balanced', random_state=25, n_jobs=4, fit_intercept=False)
t0 = time.time()
snap_svm.fit(X_train, y_train)
snap_speed = time.time() - t0
print(f"Snap Ml training time(s): {snap_speed:.2f}")

# Evaluating model speed and accuracy
training_speedup = sk_speed / snap_speed
print(f"[Training time speedup] Snap ML vs Scikit-Learn: {training_speedup}")

sklearn_pred = sk_svm.predict(X_test)
snapml_pred = snap_svm.predict(X_test)
print(f"[Scikit-Learn Confusion Matrix: \n{confusion_matrix(y_test, sklearn_pred)}")
print(f"[Snap ML Confusion Matrix: \n{confusion_matrix(y_test, snapml_pred)}")
print(f"[Scikit-Learn Classification Report: \n{classification_report(y_test, sklearn_pred)}")
print(f"[Snap ML Classification Report: \n{classification_report(y_test, snapml_pred)}")



















