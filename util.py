import numpy as np

from pandas import DataFrame
from sklearn.utils import shuffle


def balance(data: DataFrame, y_col='Control', alpha=2):
    """
    Balance the data so both classes are equally represented
    :param data: The data to balance
    :param y_col: The name of the column that stores labels
    :param alpha: Constant that determines ratio between classes in data balance (2 is equal representation)
    :return:
    """
    pos_samples = sum(data[y_col].as_matrix())
    data = data.sort_values(by=y_col, ascending=pos_samples > len(data))
    data = data.iloc[:(pos_samples * alpha)]
    return shuffle(data)


def normalize(xs):
    """
    Normalize the data by subtracting the mean and dividing the standard deviation
    :param xs: The data samples to be normalized
    :return: Normalized data
    """
    mean = np.mean(xs)
    std = np.std(xs)
    return (xs - mean) / std


