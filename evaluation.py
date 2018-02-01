from metrics import confusion_matrix


class Evaluation:
    """
    Stores the results of an evaluation run and prints a pretty table of the results
    """
    def __init__(self, y_true, y_pred, metrics):
        """
        Create a new Evaluation
        :param y_true: The true labels
        :param y_pred: The predicted labels
        :param metrics: The metrics by which the run should be evaluated
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.metrics = metrics

    def __str__(self):
        """
        :return: A pretty table containing the performance metrics
        """
        s = confusion_matrix(self.y_true, self.y_pred)
        s += '\n-------------------------+----------+\n'
        for f in self.metrics:
            s += '{:25.25}|{:10.5f}|\n'.format(f.__name__, f(self.y_true, self.y_pred))
            s += '-------------------------+----------+\n'
        return s
