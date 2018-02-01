import numpy as np

from evaluation import Evaluation
from models import Model


def k_fold_cross_validation(data, labels, k: int, model: Model, performance_metrics):
    """
    Evaluate a model by k-fold cross validation
    :param data: The data to train the model on
    :param labels: The labels corresponding to the data
    :param k: Number of folds
    :param model: The model to be evaluated
    :param performance_metrics: A list of performance metrics with which the model should be evaluated
    """
    if len(data) != len(labels):
        raise Exception('Data size does not match label size!')

    # Make list length divisible by k
    data, labels = data[len(data) % k:], labels[len(labels) % k:]
    # Determine size of each sample
    sample_size = int(len(data) / k)
    # Create k samples of equal size
    data_folds = [data[i:(i + sample_size)] for i in range(0, len(data), sample_size)]
    labels_folds = [labels[i:(i + sample_size)] for i in range(0, len(labels), sample_size)]
    # Iterate through folds
    for fold in range(k):
        # Separate validation / train samples
        validation_d, validation_l = data_folds[fold], labels_folds[fold]
        data_fold, labels_fold = [], []
        for sample in [data_folds[i] for i in range(k) if i != fold]:
            data_fold.extend(sample)
        for sample in [labels_folds[i] for i in range(k) if i != fold]:
            labels_fold.extend(sample)
        data_fold, labels_fold = np.array(data_fold), np.array(labels_fold)
        # Perform training
        model.fit(data_fold, labels_fold)
        # Perform classification
        classifications = model.predict(validation_d)
        classifications = np.round(classifications)
        # Determine performance
        print(type(model).__name__)
        print('Fold {}'.format(fold))
        e = Evaluation(validation_l, classifications, performance_metrics)
        print(e)


def hold_out_validation(data, labels, f: float, model: Model, performance_metrics):
    """
    Evaluate the model by holding out part of the data set for validation
    :param data: The data to train the model on
    :param labels: The labels corresponding to the data
    :param f: The ratio between train/test data. f is the train data percentage
    :param model: The model to be evaluated
    :param performance_metrics: A list of performance metrics with which the model should be evaluated
    """
    if len(data) != len(labels):
        raise Exception('Data size does not match label size!')

    # Compute the index where the data should be split
    split_index = int(f * len(data))
    # Split the data
    train_data, test_data = data[:split_index], data[split_index:]
    train_labels, test_labels = labels[:split_index], labels[split_index:]
    # Fit the model on the train data
    model.fit(train_data, train_labels)
    # Classify the test data
    classifications = model.predict(test_data)
    classifications = np.round(classifications)
    # Use the test labels to compute each performance metric
    print(type(model).__name__)
    e = Evaluation(test_labels, classifications, performance_metrics)
    print(e)
