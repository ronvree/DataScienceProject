import numpy as np
import keras as ks

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class Model:
    """
        Wrapper class for the models to be tested
    """

    def fit(self, train_data, train_labels):
        raise Exception('Unimplemented!')

    def predict(self, samples):
        raise Exception('Unimplemented!')


class SNN(Model):
    """
    Shallow Neural Network classifier
    """
    def __init__(self, nodes, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Dense(nodes, activation='relu', input_shape=input_shape))
        self.model.add(ks.layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_samples):
        self.model.fit(train_data, train_samples, batch_size=32, epochs=32, verbose=2)

    def predict(self, samples):
        return self.model.predict(samples)


class CNN(Model):
    """
    Convolutional Neural Network classifier
    """
    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same',
                                        input_shape=input_shape + (1,)))
        self.model.add(ks.layers.MaxPooling1D(2))
        self.model.add(ks.layers.Conv1D(32, kernel_size=5, activation='relu'))
        self.model.add(ks.layers.MaxPooling1D(2))
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(16, activation='relu'))
        self.model.add(ks.layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def fit(self, train_data, train_samples):
        self.model.fit(train_data.reshape(train_data.shape + (1,)), train_samples, batch_size=32, epochs=32, verbose=2)

    def predict(self, samples):
        return self.model.predict(samples.reshape(samples.shape + (1,)))


class RawCNN(Model):
    """
    Convolutional Neural Network classifier for the raw data
    """
    def __init__(self, input_shape):
        self.model = ks.Sequential()
        self.model.add(ks.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same',
                                        input_shape=input_shape + (1,)))
        self.model.add(ks.layers.MaxPooling1D(2))
        self.model.add(ks.layers.Conv1D(32, kernel_size=5, activation='relu'))
        self.model.add(ks.layers.MaxPooling1D(2))
        self.model.add(ks.layers.Flatten())
        self.model.add(ks.layers.Dense(32, activation='relu'))
        self.model.add(ks.layers.Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

        print(self.model.summary())

    def fit(self, train_data, train_samples):
        self.model.fit(train_data.reshape(train_data.shape + (1,)), train_samples, batch_size=32, epochs=10, verbose=2)

    def predict(self, samples):
        return self.model.predict(samples.reshape(samples.shape + (1,)))


def dtw_distance(ts_a, ts_b, w, d=lambda x, y: abs(x - y)):
    """
    Dynamic Time Warping similarity metric between two time series
    :param ts_a: Time Series A
    :param ts_b: Time Series B
    :param w: Window parameter
    :param d: Distance metric between two data points in the series
    :return: The DTW similarity between the signals
    """
    ts_a, ts_b = np.array(ts_a), np.array(ts_b)
    m, n = len(ts_a), len(ts_b)
    cost = np.math.inf * np.ones((m, n))

    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, m):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, n):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    for i in range(1, m):
        for j in range(max(1, i - w),
                       min(n, i + w)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])
    return cost[-1, -1]


def distance_matrix(xss, yss, w=10):
    """
    Create a distance matrix using DTW between the time series
    :param xss: List of time series x
    :param yss: List of time series y
    :param w: Window parameter for DTW
    :return: A distance matrix
    """
    x_shape, y_shape = xss.shape, yss.shape
    dm = np.zeros((x_shape[0], y_shape[0]))

    for i in range(x_shape[0]):
        print(i / x_shape[0])
        for j in range(y_shape[0]):
            dm[i, j] = dtw_distance(xss[i], yss[j], w)
    return dm


class KnnDtw(Model):
    """
    K-Nearest-Neighbour Classifier using DTW as distance metric
    """
    def __init__(self):
        self.fit_data = None
        self.fit_labels = None

    def fit(self, train_data, train_labels):
        self.fit_data = train_data[::100]
        self.fit_labels = train_labels[::100]

    def predict(self, samples):
        dm = distance_matrix(samples, self.fit_data, w=10)
        n_neighbours = 1
        knn_idx = dm.argsort()[:, :n_neighbours]
        knn_labels = self.fit_labels[knn_idx]
        return np.ravel(knn_labels)


class GaussianNaiveBayes(Model):
    """
    Naive Bayes Classifier
    """
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)


class DecisionTree(Model):
    """
    Decision Tree Classifier
    """
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion='gini')

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)


class RandomForest(Model):
    """
    Random Forest Classifier
    """
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=10, criterion='gini')

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)


class KNNClassifier(Model):
    """
    K-Nearest-Neighbour Classifier
    """
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)


class GradientBoosting(Model):
    """
    Gradient Boosting Classifier
    """
    def __init__(self):
        self.model = GradientBoostingClassifier()

    def fit(self, train_data, train_labels):
        self.model.fit(train_data, train_labels)

    def predict(self, samples):
        return self.model.predict(samples)
