# Credit Card Fraud Detection with Decision Trees and Support Vector Machines

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Visualizing the Data](#visualizing-the-data)
- [Contributions](#contributions)
- [Acknowledgments](#acknowledgments)
- [Change Log](#change-log)
- [License](#license)

## Introduction
This repository contains a machine learning project focused on detecting credit card fraud using Decision Tree and Support Vector Machine (SVM) classifiers. The project compares the performance of models built with scikit-learn and Snap ML, particularly in terms of training speed and accuracy. Python and popular data science libraries such as scikit-learn, pandas, matplotlib, and Snap ML are utilized in this project.

## Objectives
The primary objectives of this project are:
- To implement and compare Decision Tree classifiers using scikit-learn and Snap ML.
- To implement and compare Support Vector Machine classifiers using scikit-learn and Snap ML.
- To evaluate the performance of these models on a large-scale credit card fraud dataset.
- To visualize class distributions and model performance metrics.

## Dataset
The dataset used in this project, `creditcard.csv`, contains data on credit card transactions labeled as fraudulent or non-fraudulent. The dataset includes the following columns:
- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1` to `V28`: The result of a PCA transformation applied to the original features (due to confidentiality).
- `Amount`: The transaction amount.
- `Class`: Label where 1 indicates fraud and 0 indicates non-fraud.

[Dataset Source](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv)

### Requirements

Ensure you have the following dependencies installed:

- `Python 3.x`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn==1.0.2`
- `snapml`

## Installation
To run this project locally, you need to have Python installed along with the required libraries. You can install the necessary packages using the following command:

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/idaraabasiudoh/drug-classification-decision-tree.git
cd creditcard-fraud-detection
pip install -r requirements.txt
```
## Usage
To use this repository, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/idaraabasiudoh/drug-classification-decision-tree.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd creditcard-fraud-detection
    ```
3. **Run the classification script**:
    ```bash
    python classify_fraud.py
    ```

## Modeling
The modeling process involves the following steps:

1. **Data Loading**: Import the dataset and inflate it to increase the size for training.
2. **Data Preprocessing**: Standardize the features and normalize the data.
3. **Data Splitting**: Split the data into training and testing sets.
4. **Model Training**: Train both Decision Tree and Support Vector Machine models using scikit-learn and Snap ML.
5. **Model Prediction**: Make predictions on the test set using the trained models.

## Evaluation
The performance of the models is evaluated using the test dataset. The key metrics used for evaluation include:

- **Confusion Matrix**: A table used to describe the performance of a classification model.
- **Classification Report**: This includes precision, recall, F1-score, and support for each class.
- **Training Time Speedup**: A comparison of the training times between scikit-learn and Snap ML.

### Example Code
Here is an example of how to evaluate the models using a confusion matrix and classification report:

```python
from sklearn.metrics import confusion_matrix, classification_report

# Assuming y_test contains the actual values and pred contains the predicted values
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))
```

## Visualizing the Data
The target class distribution can be visualized using a pie chart. The following code snippet demonstrates how to create this visualization:

```python
import matplotlib.pyplot as plt

# Plotting class column to visualize data
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Target Class Values")
plt.show()
```

## Contributions
We welcome contributions from the community to improve this project. To contribute, please follow these steps:

1. **Fork the Repository**: Click the "Fork" button at the top right of the repository page to create a copy of this repository on your GitHub account.

2. **Clone the Repository**: Clone your forked repository to your local machine.
    ```bash
    git clone https://github.com/idaraabsiudoh/creditcard-fraud-detection.git
    ```

3. **Create a New Branch**: Create a new branch for your feature or bug fix.
    ```bash
    git checkout -b feature-name
    ```

4. **Make Changes**: Make your changes to the codebase.

5. **Commit Your Changes**: Commit your changes with a clear and descriptive commit message.
    ```bash
    git commit -m "Description of your changes"
    ```

6. **Push to Your Branch**: Push your changes to your forked repository.
    ```bash
    git push origin feature-name
    ```

7. **Open a Pull Request**: Open a pull request to merge your changes into the main repository. Provide a detailed description of your changes in the pull request.

We appreciate your contributions and will review your pull request as soon as possible. Thank you for helping to improve this project!

## Acknowledgments
We would like to acknowledge the following individuals for their contributions and support in the development of this project:

- **[Idara-Abasi Udoh](http://www.linkedin.com/in/idaraabasiudoh)**
- **[Andreea Anghel]**
- **[Joseph Santarcangelo]**

## Change Log

| Date (YYYY-MM-DD) | Version | Changed By        | Change Description                               |
| ----------------- | ------- | ----------------- | ------------------------------------------------ |
| 2024-08-19        | 1.0     | Idara-Abasi Udoh  | Project completion                         |

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
