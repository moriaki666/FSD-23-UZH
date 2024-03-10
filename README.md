# FSD-23-University of Zürich
Foundations of Data Science - University of Zürich

The target groups for this course are the MSc students (with major or minor in data science, or studying computer science) and PhD students across the university. This is the main coourse for Data Science Majors

## Course Topics


Introduction to Machine Learning
What is Machine Learning? Programming vs Learning; Evolution of Machine Learning; Machine Learning in Action
An Early Automatic Classification Example and The Perceptron Algorithm
Models and Methods Covered in This Lecture
Applications: House Price Prediction; Object Detection and Classification
Learning Flavours: Supervised (Regression, Classification); Unsupervised (Dimensionality Reduction, Clustering); Active; Semi-Supervised; Collaborative Filtering; Reinforcement Learning
Mathematics for Machine Learning
Linear Algebra: Vectors (Vector Norms, Inner Product Spaces); Matrices (Operations); Eigenvectors and Eigenvalues; Positive (Semi-)Definiteness
Calculus: Continuous and Differentiable Functions of One or Multiple Variables; Finding Extrema (First Derivative Test, Second Derivative Test, Critical Points); Partial Derivatives, Gradients, Hessian, Jacobian; Matrix Calculus, Chain Rule in Higher Dimensions; Optimality with Side Conditions, Lagrange Multipliers
Probability Theory: Probability Space; Conditional Probability, Bayes Rule; Random Variables; Joint Probability Distributions; Expectation, Variance, Standard Deviation, Covariance; Discrete and Continuous Probability Disctributions (Bernoulli, Binomial, Multivariate Normal, Laplace)
Linear Regression
Linear Regression by Example
Definition, Bias, Noise, One-hot Encoding, Learning vs Testing
Least Squares Objective, Gradient, Computing the Paramters
Finding Optimal Solution using Matrix Calculus, Differentiating Matrix Expressions, Deriving the Least Squares Estimates
Computational Complexity of Parameter Estimation
Least Squares Estimate in the Presence of Outlliers
Maximum Likelihood
Probabilistic vs Optimisation Views in Machine Learning
Maximum Likelihood Principle, Examples
Probabilistic Formulation of the Linear Model via Maximum Likelihood, Gaussian Noise Model
Maximum Likelihood Estimator vs Least Squares Estimator, (Log, Negative Log) Likelihood of Linear Regression
Outliers, Maximum Likelihood for Laplace Noise Model
Basis Expansion, Learning Curves, Overfitting, Validation
Basis Expansion: Polynomial, Radial Basis, using Kernels, Kernel Trick
How to Choose Hyperparameters for RBF Kernel?
Generalisation Error, Bias-Variance Tradeoff, Learning Curves
Overfitting: How Does it Occur? How to Avoid it?
Validation Error, Training and Validation Curves, Overfitting on the Validation Dataset (Kaggle Learderboard)
k-fold Cross-Validation, Grid Search
Regularisation
Estimate for Ridge Linear Regression, Lagrangian (Constrained Optimisation) Formulation
LASSO: Least Absolute Shrinkage and Selection Operator
Effect of Ridge and Lasso Hyperparameter on Weights
Feature Selection
Goal, Premise, and Motivation
Feature Selection to Reduce Overfitting
Feature Selection Methods: Wrapper methods (Forward Stepwise Selection), Filter methods (Mutual Information, Pearson Correlation Coefficient), Embedded methods (LASSO, Elastic Net Regularisation)
Convex Optimisation
Convex Sets, Examples, Proving Common Cases of Convex Sets (PSD Cone, Norm Balls, Polyhedra)
Convex Functions, Examples
Convex Optimisation Problems: Classes (Linear Programming, Quadratically Constrained Quadratic Programming), Local vs Global Optima, Proof of Local=Global Theorem
Examples: Linear Model with Absolute Loss, Minimising the Lasso Objective, Linear Regression with Gaussian Noise Model
First-Order and Second-Order Optimisation
Calculus Background: Gradient Vectors, Contour Curves, Direction of Steepest Increase, Sub-gradient, Hessian
Gradient Descent: Algorithm, Geometric Interpretation, Choosing Step Size (Backtracking Line Search), Convergence Test, Stochastic vs (Mini-)Batch, Sub-gradient Descent
Constrained Convex Optimisation: Projected Ggradient Descent
Newton’s Method: second-order Taylor Function Approximation, Geometric Interpretation, Computation and Convergence
Generative Models for Classification
Discriminative vs Generative Models
Supervised Learning: Regression vs Classification
Generative Classification Model: Definition, Prediction
Maximum Log-Likelihood Estimator for Class Probability Distribution
Naïve Bayes Classifier: Training and Predicting with Missing Data
Gaussian Discriminant Analysis: Maximum Likelihood Estimator, Quadratic/Linear Discriminant Analysis, Two-Class Linear Discriminant Analysis, Decision Boundaries, Sigmoid and Softmax Functions
Logistic Regression
Models for Binary Classification
Logistic Regression: Definition, Prediction, Contour Lines Represent Class Label Probabilities, Negative Log-Likelihood vs Cross-Entropy, Maximum Likelihood Estimate, Newton Method for Optimising the Negative Log-Likelihood, Iteratively Re-Weighted Least Squares
Multiclass Classification
One-vs-One, One-vs-Rest, Error Correcting Approach
Softmax, Multiclass Logistic Regression
Measuring Performance for Classification
Confusion Matrix, Sensitivity, Recall, Specificity, Precision, Accuracy; Examples
ROC (Receiver Operating Characteristic) Curve, Confusion Matrices for Different Decision Boundaries, Area under the ROC Curve
Precision-Recall Curve
Support Vector Machines
Maximum Margin Principle, Support Vectors, Formulation as Convex Optimisation Problem in the Linearly (Non-)Separable Case
Hinge Loss Optimisation, Hinge vs Logistic
Primal vs Dual Formulation, Constrained Optimisation with Inequalities, Karush-Kuhn-Tucker Conditions, When to Prefer the Dual Formulation
Kernel Methods: Mercer Kernels in SVM Dual Formulation, Kernel Engineering, Examples with Polynomial, RBF, and String Kernels
Neural Networks
Multi-layer Perceptrons: Example, Matrix Notation, Multi-layer Perceptron vs Logistic Regression
The Backpropagation Algorithm: Example, Forward and Backward Equations, Computational Aspects
Training Neural Networks: Difficulties (Saturation, Vanishing Gradient, Overfitting), Known Hacks (Early Stopping, Adding Data, Dropout), Rectified Linear Unit, Dying ReLU, Leaky ReLU, Initialising Weights and Biases, Examples
Convolutional Neural Networks: Convolution, Pattern-Detecting Filters, Convolutional Layers, Pooling, Popular Convolutional Neural Networks, Training
Clustering
Clustering Objective
k-Means Clustering: Algorithm, Convergence, Choosing k,
Transforming input formats: Euclidean Space, Dissimilarity Matrix, Singular Value Decomposition, Multidimensional Scaling
Hierarchical Clustering: Linkage Algorithms
Spectral Clustering
Principal Component Analysis
Dimensionality Reduction
Maximum Variance View vs Best Reconstruction View of Principal Component Analysis
Finding Principal Components using Singular Value Decomposition and Iterative Method
Applications: Reconstruction of an Image using PCA, Eigenfaces, Latent Semantic Analysis
Practicals

The practical tasks require implementation using jupyter notebooks, Python, Scikit-learn, and TensorFlow.

Implementation of Linear Regression (Ridge, Lasso)

Comparison of Generative and Discriminative Models
