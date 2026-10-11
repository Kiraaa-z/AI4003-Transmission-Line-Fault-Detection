# Detecting Three-Phase Transmission-Line Faults: Logistic Regression vs. Support Vector Machines

## Introduction

Electrical faults in transmission lines can affect power-system reliability and require prompt detection. Machine learning offers a way to investigate whether electrical measurements, such as phase currents and voltages, can help distinguish normal operating conditions from fault conditions.

In this project, we compared logistic regression implemented using NumPy with Support Vector Machines (SVMs) to classify transmission-line measurements into two categories: fault and no fault. We also investigated whether adding squared features could improve a linear logistic-regression model and compared linear and nonlinear SVM decision boundaries.

The aim was not simply to obtain a high accuracy score, but to examine how feature engineering, model selection, and evaluation choices affect fault-detection performance.

## Dataset and Data Preparation

The dataset contained 12,001 rows with six electrical measurements: three-phase currents (Ia, Ib, and Ic) and three-phase voltages (Va, Vb, and Vc). The target variable, `Output (S)`, represented the class label, where 0 indicated no fault and 1 indicated a fault.

The original dataset also contained two completely empty columns. These were removed during cleaning, leaving the six input features and the target label. The dataset contained 6,505 no-fault samples and 5,496 fault samples.

We divided the data into training, validation, and test sets using a 60:20:20 split with random seed 42. This produced 7,200 training rows, 2,400 validation rows, and 2,401 test rows. Standardisation parameters were calculated from the training set and then applied to the other splits to avoid using validation or test information during preprocessing.

The validation set was used to compare model configurations, while the test set was reserved for the final evaluation.

## Exploratory Data Analysis

We used three plots to explore the electrical measurements and their relationship with the fault label. The first plot shows the three phase currents over the first 2,000 rows, with fault-labelled regions shaded. It illustrates how the current signals can change substantially during a fault-labelled interval.

The second plot compares Ia and Ib, with points coloured by class. The two classes overlap, indicating that a simple straight-line boundary may not separate fault and no-fault samples effectively. The third plot compares the Ia distributions for both classes. Their overlap further suggests that a single current measurement is not sufficient for reliable classification.

These observations motivated the comparison of linear models with approaches that can represent nonlinear relationships.

## Models and Feature Engineering

We first implemented logistic regression from scratch using NumPy and batch gradient descent. The model learns weights for the input features and uses the sigmoid function to estimate the probability of a fault. We tested learning rates of 0.01, 0.1, 1, and 10, selecting a rate based on the cost curve.

Two feature sets were compared. Set R contains the six standardised current and voltage measurements. Set Q adds the square of each measurement, giving the model 12 input features plus an intercept. Although logistic regression remains linear in its learned weights, the squared terms allow it to represent a nonlinear decision boundary in the original measurement space.

We also trained Support Vector Machines using a linear kernel and an RBF kernel. The linear SVM provides a linear decision boundary, whereas the RBF kernel can model more complex relationships. We selected the configurations using validation accuracy rather than the test set, helping preserve the test set for final evaluation.

## Results and Discussion

The final evaluation showed a clear difference between the linear models and the models using nonlinear features or kernels. The table below summarises the results on the held-out test set.

| Model | Test accuracy | False alarms | Missed faults |
|---|---:|---:|---:|
| Logistic Regression — Set R | 72.64% | 0 | 657 |
| Logistic Regression — Set Q | 97.88% | 6 | 45 |
| Linear SVM | 73.59% | 0 | 634 |
| RBF SVM | 99.58% | 4 | 6 |

Set R logistic regression and the linear SVM achieved similar accuracy, at 72.64% and 73.59%, respectively. This is consistent with both models learning linear decision boundaries from the same six electrical measurements.

Adding squared features produced a substantial improvement. Set Q logistic regression achieved 97.88% test accuracy, with 45 missed faults compared with 657 for Set R. This suggests that the squared terms captured useful nonlinear patterns in the measurements.

The RBF SVM achieved the highest test accuracy, at 99.58%, and missed only six faults. It also produced four false alarms. For fault detection, accuracy alone is not sufficient: a missed fault and a false alarm have different operational consequences, so both must be considered when evaluating a model.

## SVM Validation Results

We compared the SVM configurations using validation accuracy, varying the regularisation parameter C and, for the RBF kernel, the gamma parameter. The linear SVM achieved 72.58% validation accuracy for all tested values of C. In contrast, the RBF SVM achieved 99.67% at C = 100 and gamma = 1, making it the best configuration among those tested.

The validation plot illustrates the difference between the linear and nonlinear kernels. It also shows why model selection should use validation data rather than repeatedly adjusting configurations based on test results.

## Limitations and Conclusion

Although the RBF SVM performed best on the test set, the results should be interpreted cautiously. A random split can place neighbouring time samples in different subsets, potentially making the test set less independent than data collected during a separate operating period. Further evaluation on unseen operating conditions is therefore necessary.

The cost of a missed fault may be much greater than the cost of a false alarm. For this reason, a practical protection system would require careful threshold selection, independent testing, response-time evaluation, and validation by power-system engineers. These models are experimental classifiers and should not be used as replacements for validated protection equipment.

Overall, the experiments showed that feature engineering substantially improved logistic regression, while the RBF SVM achieved the strongest classification results among the tested models. The project demonstrates why model evaluation should consider not only accuracy, but also missed faults, false alarms, data preparation, and generalisation to new conditions.