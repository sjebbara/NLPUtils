import sklearn

import numpy


def accuracy(target_labels, predicted_labels):
    correct = float(sum([t == p for t, p in zip(target_labels, predicted_labels)]))
    incorrect = float(len(target_labels) - correct)
    acc = correct / (correct + incorrect)
    return acc, correct, incorrect


def logloss(Y_true, Y_pred_probas):
    epsilon = 10e-10
    m = Y_true.shape[0]
    Y_logs = numpy.log(numpy.maximum(numpy.minimum(Y_pred_probas, 1 - epsilon), epsilon))
    prod = numpy.multiply(Y_true, Y_logs, dtype=float)
    return -numpy.sum(prod) / m


def tp_fp_fn(targets, predictions, matcher=None):
    if type(targets) is not set:
        targets = set(targets)
    if type(predictions) is not set:
        predictions = set(predictions)

    if matcher:
        tp = sum([any([matcher(t, p) for t in targets]) for p in predictions])
    else:
        tp = float(len(targets & predictions))
    fp = len(predictions) - tp
    fn = len(targets) - tp

    # tp = float(len(targets & predictions))
    # fp = float(len(predictions - targets))
    # fn = float(len(targets - predictions))
    return tp, fp, fn


def precision_recall(tp, fp, fn):
    # if tp == 0 and fp == 0:
    #     precision = 1
    # else:
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    try:
        precision = tp / (tp + fp)  # == TP/len(predictions)
    except ZeroDivisionError as e:
        precision = float("nan")

    # if tp == 0 and fn == 0:
    #     recall = 1
    # else:
    try:
        recall = tp / (tp + fn)  # == TP/len(targets)
    except ZeroDivisionError as e:
        recall = float("nan")
    return precision, recall


def f1(beta=1, p=None, r=None, tp=None, fp=None, fn=None, targets=None, predictions=None):
    if targets is not None and predictions is not None:
        tp, fp, fn = tp_fp_fn(targets=targets, predictions=predictions)
    if tp is not None and fp is not None and fn is not None:
        p, r = precision_recall(tp=tp, fp=fp, fn=fn)
    if p is not None and r is not None:
        try:
            fscore = (1 + beta ** 2) * r * p / (r + (beta ** 2 * p))
        except ZeroDivisionError as e:
            fscore = float("nan")
        return fscore, p, r
    return


def g1(p=None, r=None, tp=None, fp=None, fn=None, targets=None, predictions=None):
    if targets is not None and predictions is not None:
        tp, fp, fn = tp_fp_fn(targets=targets, predictions=predictions)
    if tp is not None and fp is not None and fn is not None:
        p, r = precision_recall(tp=tp, fp=fp, fn=fn)
    if p is not None and r is not None:
        return numpy.sqrt(r * p), p, r


def semeval_f1(correct, predicted, b=1):
    common, relevant, retrieved = 0., 0., 0.
    for i in range(len(correct)):
        cor = [a for a in correct[i]]
        pre = [a for a in predicted[i]]
        common += len([a for a in pre if a in cor])
        retrieved += len(pre)
        relevant += len(cor)
    p = common / retrieved if retrieved > 0 else 0.
    r = common / relevant
    f1 = (1 + (b ** 2)) * p * r / ((p * b ** 2) + r) if p > 0 and r > 0 else 0.
    return p, r, f1, common, retrieved, relevant

def auc(target_labels, predicted_scores):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(target_labels, predicted_scores)
    auc = sklearn.metrics.auc(fpr, tpr)
    return auc
