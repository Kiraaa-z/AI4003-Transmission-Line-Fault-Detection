# Task 6 — Model Analysis

## 6.1 Logistic Regression Set R vs. Linear SVM
Logistic regression on Set R achieved 72.64% test accuracy, while the linear SVM achieved 73.59%. Their performance is similar because both models use the same six standardised electrical measurements and learn a linear decision boundary. This suggests that a linear boundary may not capture the nonlinear patterns associated with transmission-line faults effectively.

## 6.2 Signs of the Set Q Squared-Feature Weights
The Set Q model includes squared current and voltage features. The learned weights for the squared current terms (Ia², Ib², and Ic²) are positive, while the weights for the squared voltage terms (Va², Vb², and Vc²) are negative. This suggests that, in the fitted model, larger squared current measurements contribute toward predicting a fault, whereas larger squared voltage measurements contribute toward predicting no fault. These relationships are consistent with the idea that faults can cause current to increase and voltage to decrease.

## 6.3 Comparing Support Vectors
The best linear SVM (C = 0.1) used 5,847 support vectors, while the best RBF SVM (C = 100, γ = 1) used 274. The linear SVM's large support-vector count indicates that many training samples influence its decision boundary. The RBF SVM achieves much higher validation accuracy with fewer support vectors, suggesting that its nonlinear decision boundary fits the dataset's fault patterns more effectively.

## 6.4 Effect of C on Accuracy and Support Vectors
For the linear SVM, validation accuracy remained approximately 72.58% as C increased from 0.1 to 100, while the number of support vectors decreased slightly from 5,847 to 5,834. For the RBF SVM, accuracy generally improved as C increased, reaching 99.67% at C = 100 and γ = 1. However, the support-vector count did not decrease consistently because it also depends on γ. The parameter C controls the penalty for classification errors: a larger C penalises training errors more strongly and can produce a more complex decision boundary.

## 6.5 Random Split and Real-World Performance
The test accuracy may be optimistic for a real transmission-line protection system because the dataset contains consecutive time samples, and a random split can place neighbouring, highly similar samples in both training and test sets. The models may therefore be evaluated on patterns that are very similar to those they have already learned. A chronological split or evaluation on data from a different operating period would provide a stronger test of generalisation to unseen conditions.

## 6.6 Model Selection for a Protection Relay
Based on the current test results, the RBF SVM is the strongest candidate for further evaluation because it achieved 99.58% test accuracy, with only 6 missed faults and 4 false alarms. In comparison, Set Q logistic regression missed 45 faults and produced 6 false alarms, while Set R logistic regression missed 657 faults. However, these results alone are not enough to deploy the RBF SVM in a real protection relay. It would require further testing on independent operating conditions, and its reliability, response time, and safety performance would need to be validated before practical use.