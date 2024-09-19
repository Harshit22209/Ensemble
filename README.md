# Ensemble Methods: AdaBoost and Gradient Boosting on MNIST

This project demonstrates the application of **AdaBoost** and **Gradient Boosting** on the MNIST dataset. The aim is to show how these ensemble methods perform on binary and multi-class classification as well as regression tasks. The dataset is pre-processed using PCA to reduce dimensionality, and weak learners (decision stumps) are employed to build the models.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Methods](#methods)
   - [AdaBoost (Binary Classification)](#adaboost-binary-classification)
   - [Gradient Boosting (Regression)](#gradient-boosting-regression)
   - [Decision Tree & Bagging (Multi-class Classification)](#decision-tree--bagging-multi-class-classification)
4. [Results](#results)
   - [Accuracy and MSE](#accuracy-and-mse)
5. [Conclusion](#conclusion)
6. [Future Work](#future-work)
7. [References](#references)

---

## Project Overview

This project focuses on implementing two key ensemble methods:
- **AdaBoost.M1** for binary classification of digits 0 and 1 (from MNIST).
- **Gradient Boosting** for regression using the same dataset.
- **Bagging** and decision tree for multi-class classification on digits 0, 1, and 2.

The project shows the ability of these techniques to handle both classification and regression tasks using weak learners (decision stumps), along with PCA for dimensionality reduction.

---

## Dataset

We used the [MNIST dataset](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) in this project. The dataset was processed as follows:
- **Binary Classification**: Filtered for digits 0 and 1, labeled as -1 and 1.
- **Multi-class Classification**: Filtered for digits 0, 1, and 2.
- **PCA**: Applied Principal Component Analysis to reduce the dataset's dimensionality to 5 or 10 dimensions, depending on the task.

---

## Methods

### AdaBoost (Binary Classification)
For binary classification between digits 0 and 1:
1. **Data Preparation**: PCA reduced data to 5 dimensions.
2. **Decision Stumps**: Weak learners were created by minimizing weighted misclassification error.
3. **AdaBoost Algorithm**:
   - Updated sample weights after each iteration based on classification errors.
   - Built 300 decision stumps.
4. **Evaluation**:
   - Reported accuracy on both validation and test sets.
   - Plotted accuracy vs. number of trees.

### Gradient Boosting (Regression)
For regression on digits 0 and 1 using absolute loss:
1. **Data Preparation**: PCA reduced data to 5 dimensions.
2. **Gradient Boosting**:
   - Decision stumps were trained by minimizing Sum of Squared Residuals (SSR).
   - Updated labels with negative gradients in each iteration.
   - Trained 300 decision stumps.
3. **Evaluation**:
   - Reported MSE on validation and test sets.
   - Plotted MSE vs. number of iterations.

### Decision Tree & Bagging (Multi-class Classification)
For multi-class classification (digits 0, 1, and 2):
1. **Data Preparation**: PCA reduced data to 10 dimensions.
2. **Decision Tree**:
   - Built a decision tree with 3 terminal nodes using Gini index to find optimal splits.
   - Assigned classes based on the majority in segmented regions.
3. **Bagging**:
   - Created 5 datasets using bootstrapping and built trees for each.
   - Used majority voting for predictions (at least 3 trees must agree).

---

## Results

### Accuracy and MSE

- **AdaBoost**:
   - Best validation accuracy: **99.75%** at 65 iterations.
   - Test accuracy: **99.71%** after 161 iterations.
- **Gradient Boosting**:
   - MSE dropped from **0.5 to 0.12** over 300 iterations.
   - Noted overfitting beyond 2500 iterations.
- **Bagging** (Multi-class):
   - Improved accuracy using ensemble voting compared to a single decision tree.

---

## Conclusion

This project demonstrates how ensemble methods like **AdaBoost** and **Gradient Boosting** can enhance performance in both classification and regression tasks. The combination of weak learners in ensemble methods shows significant improvements over individual models. The **Bagging** method further proves useful in multi-class settings by reducing the variance and improving overall accuracy.

---

## Future Work

- Expand the ensemble methods to all 10 digits of the MNIST dataset.
- Experiment with advanced ensemble techniques such as **XGBoost** and **Random Forests**.
- Explore other loss functions and regularization methods for further improvement in performance.

---

## References

- MNIST Dataset: [TensorFlow MNIST](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
- AdaBoost: "A Short Introduction to Boosting" by Yoav Freund and Robert E. Schapire
- Gradient Boosting: "Greedy Function Approximation: A Gradient Boosting Machine" by Jerome H. Friedman
