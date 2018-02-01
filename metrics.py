from sklearn import metrics

_confusion_matrix_string = '\n'
_confusion_matrix_string += ' A\\P |  0  |  1  |\n'
_confusion_matrix_string += '-----+-----+-----+\n'
_confusion_matrix_string += '  0  |{:^5}|{:^5}|\n'
_confusion_matrix_string += '-----+-----+-----+\n'
_confusion_matrix_string += '  1  |{:^5}|{:^5}|\n'
_confusion_matrix_string += '-----+-----+-----+\n'
_confusion_matrix_string += 'tn: {}\n'
_confusion_matrix_string += 'fp: {}\n'
_confusion_matrix_string += 'fn: {}\n'
_confusion_matrix_string += 'tp: {}\n'


def confusion_matrix(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return _confusion_matrix_string.format(tn, fp, fn, tp, tn, fp, fn, tp)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tn / float(tn + fp)


def positive_predicted_value(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return tp / float(tp + fp)


def error_rate(y_true, y_pred):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    return (fp + fn) / float(tn + fp + fn + tp)


if __name__ == '__main__':

    print(confusion_matrix([1, 0, 0, 1, 0], [1, 1, 0, 0, 0]))


