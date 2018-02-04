from read_data import read_preprocessed_af_data
from util import balance
import matplotlib.pyplot as plt
import numpy as np

data = balance(read_preprocessed_af_data())

Xs = data.iloc[:, :-1].as_matrix()
ys = data.iloc[:, -1].as_matrix()


def convert_sample(sample):
    new_sample = np.zeros(len(sample))
    new_sample[0] = sample[0]
    for i in range(1, len(sample)):
        new_sample[i] = new_sample[i - 1] + sample[i]
    return new_sample


for X, y in zip(Xs, ys):
    plt.title(y)
    plt.plot(convert_sample(X))
    plt.show()


