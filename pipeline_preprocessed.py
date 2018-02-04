from evaluation_protocols import k_fold_cross_validation, hold_out_validation
from metrics import error_rate, positive_predicted_value, specificity
from models import *
from read_data import read_preprocessed_af_data
from sklearn import metrics

from util import balance

# Read the data
data = balance(read_preprocessed_af_data())

# Split the data and labels and convert to numpy arrays
Xs = data.iloc[:, :-1].as_matrix()
ys = data.iloc[:, -1].as_matrix()

# List the models to be evaluated
models = [
    DecisionTree(),                    # 0 Decision Tree
    GaussianNaiveBayes(),              # 1 Gaussian Naive Bayes
    RandomForest(),                    # 2 Random Forest
    KNNClassifier(),                   # 3 K Nearest Neighbours
    # KnnDtw(),                          # 4 K Nearest Neighbours with Dynamic Time Warping
    SNN(len(Xs[0]), Xs.shape[1:]),     # 5 Shallow Neural Network
    CNN(Xs.shape[1:]),                 # 6 Convolutional Neural Network
    GradientBoosting(),                # 7 Gradient Boosting
    ]

# Choose performance metrics with which the model should be evaluated
performance_metrics = [metrics.accuracy_score,
                       metrics.precision_score,
                       metrics.recall_score,
                       metrics.matthews_corrcoef,
                       specificity,
                       positive_predicted_value,
                       error_rate,
                       ]

# Obtain the model's performance from an evaluation protocol
for model in models:
    hold_out_validation(Xs, ys, 0.6, model, performance_metrics)
    # k_fold_cross_validation(Xs, ys, 10, model, performance_metrics)
