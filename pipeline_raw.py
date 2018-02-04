from evaluation_protocols import k_fold_cross_validation, hold_out_validation
from metrics import error_rate, positive_predicted_value, specificity
from models import *
from read_data import read_af_data
from sklearn import metrics

from util import balance, normalize

# Read the data
data = balance(read_af_data(), y_col='labels')

# Split the data and labels and convert to numpy arrays
Xs = data['samples'].as_matrix()
ys = data['labels'].as_matrix()

# Parse data
Xs = np.array([np.vectorize(int)(x[1:-1].split(',')) for x in Xs])
# Normalize data
Xs = normalize(Xs)

# np.append(Xs, np.std(Xs))

# Choose a model
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

for model in models:
    hold_out_validation(Xs, ys, 0.6, model, performance_metrics)
    # k_fold_cross_validation(Xs, ys, 10, model, performance_metrics)
