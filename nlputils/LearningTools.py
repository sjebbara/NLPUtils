import datetime
import functools
import gzip
import io
import itertools
import json
import math
import os
import pprint
import re
import time
import traceback
from collections import Counter
from functools import reduce
from typing import List, Tuple, Sequence

try:
    import matplotlib
    import matplotlib.markers
    import matplotlib.pyplot as plt
    import pylab
except ImportError:
    print("Warning: Plotting not available:")
    print(traceback.format_exc())
import numpy
import scipy
import six
from colorama import Fore
from numpy.random.mtrand import RandomState
from six import string_types
from sklearn.model_selection import ParameterSampler, ParameterGrid

from nlputils import EvaluationTools

__author__ = 'sjebbara'


class BetterDict(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__('_data', dict(*args, **kwargs))

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __getattr__(self, item):
        return self._data[item]

    def __setattr__(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data

    def clear(self):
        return self._data.clear()

    def copy(self):
        return self._data.copy()

    def has_key(self, k):
        return k in self._data

    def update(self, *args, **kwargs):
        return self._data.update(*args, **kwargs)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()


class Configuration(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        super().__setattr__('_data', dict(*args, **kwargs))
        super().__setattr__('_accessed_fields', set())

    def __getitem__(self, item):
        self._accessed_fields.add(item)
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __getattr__(self, item):
        self._accessed_fields.add(item)
        return self._data[item]

    def __setattr__(self, key, value):
        # if key == "accessed_fields":
        #     super(MyDict, self).__setattr__(key, value)
        # else:
        self._data[key] = value

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, item):
        return item in self._data

    def clear(self):
        return self._data.clear()

    def copy(self):
        return self._data.copy()

    def has_key(self, k):
        return k in self._data

    def update(self, *args, **kwargs):
        return self._data.update(*args, **kwargs)

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    @property
    def accessed_fields(self):
        return self._accessed_fields

    def clear_accessed_fields(self):
        self._accessed_fields.clear()

    def accessed_dict(self):
        d = {k: self[k] for k in self._accessed_fields}
        return Configuration(d)

    def _to_primitive_dict(self):
        d = dict()
        for k, v in self.items():
            if hasattr(v, "__name__"):
                d[k] = v.__name__
            else:
                d[k] = v
        return d

    def description(self):
        return pprint.pformat(self._data)

    def save(self, filepath, makedirs=False, accessed_only=False):
        if accessed_only:
            d = self.accessed_dict()._to_primitive_dict()
        else:
            d = self._to_primitive_dict()

        if makedirs:
            dirpath = os.path.dirname(filepath)
            os.makedirs(dirpath)

        with open(filepath, "w") as f:
            f.write(json.dumps(d, indent=4, sort_keys=True))

    @classmethod
    def load(cls, filepath):
        with open(filepath) as f:
            d = json.load(f)
        conf = cls(d)
        return conf


# class BetterDict(dict):
#     """
#     Example:
#     m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(BetterDict, self).__init__(*args, **kwargs)
#         for arg in args:
#             if isinstance(arg, dict):
#                 for k, v in arg.items():
#                     self[k] = v
#
#         if kwargs:
#             for k, v in kwargs.items():
#                 self[k] = v
#
#     def __getattr__(self, attr):
#         return self.get(attr)
#
#     def __setattr__(self, key, value):
#         self.__setitem__(key, value)
#
#     def __setitem__(self, key, value):
#         super(BetterDict, self).__setitem__(key, value)
#         self.__dict__.update({key: value})
#
#     def __delattr__(self, item):
#         self.__delitem__(item)
#
#     def __delitem__(self, key):
#         super(BetterDict, self).__delitem__(key)
#         del self.__dict__[key]

# class Configuration(dict):
#     """
#     Example:
#     m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
#     """
#
#     def __init__(self, *args, **kwargs):
#         super(Configuration, self).__init__(*args, **kwargs)
#         for arg in args:
#             if isinstance(arg, dict):
#                 for k, v in arg.items():
#                     self[k] = v
#
#         if kwargs:
#             for k, v in kwargs.items():
#                 self[k] = v
#
#     def __getattr__(self, attr):
#         return self.__dict__[attr]
#
#     def __getitem__(self, item):
#         return super(Configuration, self).__getitem__(item)
#
#     def __setattr__(self, key, value):
#         self.__setitem__(key, value)
#
#     def __setitem__(self, key, value):
#         super(Configuration, self).__setitem__(key, value)
#         self.__dict__.update({key: value})
#
#     def __delattr__(self, item):
#         self.__delitem__(item)
#
#     def __delitem__(self, key):
#         super(Configuration, self).__delitem__(key)
#         del self.__dict__[key]
#
#     def description(self):
#         return pprint.pformat(self)
#
#     def _to_primitive_dict(self):
#         d = dict()
#         for k, v in self.items():
#             if hasattr(v, "__name__"):
#                 d[k] = v.__name__
#             else:
#                 d[k] = v
#         return d
#
#     def save(self, filepath, makedirs=False):
#         d = self._to_primitive_dict()
#         if makedirs:
#             dirpath = os.path.dirname(filepath)
#             os.makedirs(dirpath)
#
#         with open(filepath, "w") as f:
#             f.write(json.dumps(d, indent=4, sort_keys=True))
#
#     @classmethod
#     def load(cls, filepath):
#         with open(filepath) as f:
#             # conf_str = f.read().strip()
#             d = json.load(f)
#         conf = Configuration(d)
#         return conf


class CorpusWrapper:
    def __init__(self):
        pass


class Evaluation:
    def __init__(self):
        self.F1 = []
        self.P = []
        self.R = []

        self.TP = []
        self.FP = []
        self.FN = []

        self.A = []
        self.correct = []
        self.incorrect = []

    def f1(self):
        abs_tp = numpy.sum(self.TP)
        abs_fp = numpy.sum(self.FP)
        abs_fn = numpy.sum(self.FN)

        return EvaluationTools.f1(tp=abs_tp, fp=abs_fp, fn=abs_fn)

    def accuracy(self):
        abs_correct = numpy.sum(self.correct)
        abs_incorrect = numpy.sum(self.incorrect)
        abs_total = abs_correct + abs_incorrect
        return float(abs_correct) / abs_total, abs_correct, abs_incorrect


class CrossValidationEvaluation:
    def __init__(self):
        self.experiments = []

    def add(self, evaluation):
        self.experiments.append(evaluation)

    def accuracy_last(self, data_sample_filter=None):
        avrg_acc = 0
        avrg_cor = 0
        avrg_inc = 0
        scores = numpy.zeros((len(self.experiments), 3))
        for i, e in enumerate(self.experiments):
            acc, cor, inc = e.accuracy_last(data_sample_filter)
            scores[i, :] = [acc, cor, inc]
            avrg_acc += acc
            avrg_cor += cor
            avrg_inc += inc

        avrg_acc /= len(self.experiments)
        avrg_cor /= len(self.experiments)
        avrg_inc /= len(self.experiments)
        return avrg_acc, avrg_cor, avrg_inc, scores

    def accuracy_best(self, data_sample_filter=None):
        scores = []
        for ex in self.experiments:
            scores.append(ex.get_scores(data_sample_filter))
        scores = numpy.array(scores)
        print(scores.shape)
        mean_accuracies = numpy.mean(scores[:, :, 0], axis=2)
        best_snapshot = numpy.argmax(mean_accuracies, axis=1)
        best_acc, best_cor, best_inc = scores[best_snapshot]
        return best_acc, best_cor, best_inc

    def print_accuracy_last(self, data_sample_filter=None):
        avrg_acc, avrg_cor, avrg_inc, scores = self.accuracy_last(data_sample_filter)
        print("Accuracy: {:.3f}; Correct: {:.3f}; Incorrect: {:.3f}".format(avrg_acc, avrg_cor, avrg_inc))
        print("MEAN: [{:.3f}, {:.3f}, {:.3f}]".format(*numpy.mean(scores, axis=0)))
        print("STD: [{:.5f}, {:.2f}, {:.2f}]".format(*numpy.std(scores, axis=0)))
        print("VAR: [{:.5f}, {:.2f}, {:.2f}]".format(*numpy.var(scores, axis=0)))
        for acc, cor, inc in scores:
            print("[{:.3f}, {:.0f}, {:.0f}]".format(acc, cor, inc))


class ExperimentEvaluation:
    def __init__(self):
        self.snapshots = []

    def add(self, evaluation):
        self.snapshots.append(evaluation)

    def get_scores(self, data_sample_filter=None):
        scores = numpy.zeros((len(self.snapshots), 3))
        for i, e in enumerate(self.snapshots):
            acc, cor, inc = e.accuracy()
            scores[i, :] = [acc, cor, inc]
        return scores

    def accuracy_best(self, data_sample_filter=None):
        scores = numpy.zeros((len(self.snapshots), 3))
        for i, e in enumerate(self.snapshots):
            acc, cor, inc = e.accuracy(data_sample_filter)
            scores[i, :] = [acc, cor, inc]

        best_snapshot_index = numpy.argmax(scores[:, 0], axis=0)
        best_acc, best_cor, best_inc = scores[best_snapshot_index, :]
        return best_acc, best_cor, best_inc

    def accuracy_last(self, data_sample_filter=None):
        best_acc, best_cor, best_inc = self.snapshots[-1].accuracy(data_sample_filter)
        return best_acc, best_cor, best_inc

    def print_accuracy(self, data_sample_filter=None):
        avrg_acc, avrg_cor, avrg_inc = self.accuracy_last(data_sample_filter)
        print("Accuracy: {:.3f}; Correct: {:.3f}; Incorrect: {:.3f}".format(avrg_acc, avrg_cor, avrg_inc))


class ExperimentSnapshotEvaluation:
    def __init__(self):
        self.data = []
        self.targets = []
        self.predictions = []

    def add(self, data_sample, target, prediction):
        self.data.append(data_sample)
        self.targets.append(target)
        self.predictions.append(prediction)

    def accuracy(self, data_sample_filter=None):
        if data_sample_filter:
            filtered_targets = []
            filtered_predictions = []
            for d, t, p in zip(self.data, self.targets, self.predictions):
                if data_sample_filter(d):
                    filtered_targets.append(t)
                    filtered_predictions.append(p)
        else:
            filtered_targets = self.targets
            filtered_predictions = self.predictions

        acc, correct, incorrect = EvaluationTools.accuracy(filtered_targets, filtered_predictions)
        return acc, correct, incorrect


class ExperimentSnapshotResults:
    def __init__(self):
        self.data_samples = []

    def add(self, data_sample):
        self.data_samples.append(data_sample)


class TrainingTimer(object):
    def __init__(self):
        pass

    def init(self, n_epochs, n_data):
        self.n_data = n_data
        self.n_epochs = n_epochs
        self.processed_samples = 0
        self.start_time = time.time()

    def time(self, e, i):
        # if self.processed_samples > 0:
        processed_samples = e * self.n_data + i
        if processed_samples > 0:
            remaining_samples = self.n_epochs * self.n_data - processed_samples
            # remaining_samples = self.n_data - i + (self.n_data * (self.n_epochs - e - 1))
            passed_time = time.time() - self.start_time
            avrg_time = passed_time / processed_samples
            remaining_time = avrg_time * remaining_samples
            p_h, p_m, p_s = seconds_to_readable_time(passed_time)
            r_h, r_m, r_s = seconds_to_readable_time(remaining_time)
            return "PASSED: %dh %dm %ds, REMAINING: %dh %dm %ds" % (p_h, p_m, p_s, r_h, r_m, r_s)
        return "PASSED: ?h ?m ?s, REMAINING: ?h ?m ?s"

    def time_now(self):
        if self.processed_samples > 0:
            remaining_samples = self.n_data - self.processed_samples
            passed_time = time.time() - self.start_time
            avrg_time = passed_time / self.processed_samples
            remaining_time = avrg_time * remaining_samples
            p_h, p_m, p_s = seconds_to_readable_time(passed_time)
            a_h, a_m, a_s = seconds_to_readable_time(avrg_time)
            r_h, r_m, r_s = seconds_to_readable_time(remaining_time)
            return "PASSED: %dh %dm %ds, REMAINING: %dh %dm %ds" % (p_h, p_m, p_s, r_h, r_m, r_s)
        return "PASSED: ?h ?m ?s, REMAINING: ?h ?m ?s"

    def process(self, n_processed_samples):
        self.processed_samples += n_processed_samples


class TrainingLogger(object):
    def init(self, n_epochs, n_data, log_file=None):
        self.n_data = n_data
        self.n_epochs = n_epochs
        self.timer = TrainingTimer()
        self.timer.init(n_epochs, n_data)
        self.log_file = log_file
        self.last_epoch = 0
        self.last_sample = 0

    def log_epoch(self, epoch):
        log(self.log_file, "Epoch {}/{}:".format(epoch + 1, self.n_epochs))

    def log_sample(self, epoch, sample, patience=None):
        skip = (epoch - self.last_epoch) * self.n_data + (sample - self.last_sample)
        if not patience or skip > patience:
            log(self.log_file, "Epoch {}/{};  Sample {}/{}\t\t{}".format(epoch + 1, self.n_epochs, sample, self.n_data,
                                                                         self.timer.time(epoch, sample)))
            self.last_epoch = epoch
            self.last_sample = sample


class ScorePlot(object):
    DEFAULT_PLOT_TYPE = "Score"

    def __init__(self, title, n_cross_validation, n_epochs, range=None, figsize=None):
        self.color_start = 0.5
        self.cmap = matplotlib.cm.get_cmap('gist_ncar')
        self.scores = dict()
        self.lines = dict()
        self.colors = dict()
        self.n_experiments = n_cross_validation
        self.n_epochs = n_epochs
        self.markers = list(matplotlib.markers.MarkerStyle.filled_markers)
        self.markers.remove("s")
        if figsize:
            self.fig = plt.figure(title, figsize=figsize)
        else:
            self.fig = plt.figure(title, figsize=(16, 9))

        self.ax = self.fig.add_subplot(1, 1, 1)
        if range:
            self.range = range
        else:
            self.range = (0, 1)

    def _get_color(self):
        x = (self.color_start + len(self.scores) * 0.41) % 0.9
        return self.cmap(x)

    def _init_plot_type(self, plot_type):
        c = self._get_color()
        self.scores[plot_type] = numpy.zeros((self.n_experiments, self.n_epochs)) * float("nan")
        self.colors[plot_type] = c
        self.lines[plot_type] = numpy.random.choice(["-", "--", "-.", ":"], 1)[0]

    def save(self, filepath):
        self.fig.set_size_inches(8, 6)
        self.fig.savefig(filepath, dpi=200)

    def add(self, n, e, score, plot_type=DEFAULT_PLOT_TYPE):
        if plot_type not in self.scores:
            self._init_plot_type(plot_type)

        if not math.isnan(score):
            self.ax.clear()

            plt.axis([0.8, self.n_epochs + 0.2, self.range[0], self.range[1]])
            plt.yticks(scipy.arange(self.range[0], self.range[1], 0.05))
            plt.grid()
            plt.ion()

            self.scores[plot_type][n, e] = score
            for pt, scores in self.scores.items():
                scores = self.scores[pt]
                color = self.colors[pt]
                for n_prime in range(self.n_experiments):
                    # only add label to first of its lines to keep legend small
                    self.ax.plot(range(1, self.n_epochs + 1), scores[n_prime, :], color=color,
                                 marker=self.markers[n_prime], label=pt if n_prime == 0 else None)
            for pt, scores in self.scores.items():
                line = self.lines[pt]
                color = self.colors[pt]
                means = numpy.nanmean(scores, axis=0)
                self.ax.plot(range(1, self.n_epochs + 1), means, color=color, linewidth=3, marker="s")

                mx = numpy.nanmax(means)
                if not numpy.isnan(mx):
                    self.ax.plot([1, self.n_epochs + 1], [mx, mx], linestyle=line, linewidth=1, color=color)
                    self.ax.text(self.n_epochs * 0.95, mx, "{:.4f}".format(mx), fontsize=13, color=color)

            plt.legend(loc=4)
            # plt.legend(loc=4, bbox_to_anchor=(0.5, -0.1))
            self.fig.tight_layout()
            # plt.draw()
            plt.show()
            plt.pause(0.0001)

    def print_scores(self, plot_type=None):
        if plot_type:
            plot_types = [plot_type]
        else:
            plot_types = self.scores.keys()

        for pt in plot_types:
            print("#####", pt, "#####")
            print(list(zip(numpy.argmax(self.scores[pt], axis=1), numpy.max(self.scores[pt], axis=1))))
            print(self.scores[pt])


class AttentionPlot:
    MATRIX_STYLE = "matrix"
    VECTOR_STYLE = "vector"

    class AttendedDataSample:
        def __init__(self):
            self.sequence1 = None
            self.sequence2 = None
            self.attention = None

    def __init__(self, title, data, styles, additional_information_callback=None):
        self.fig = plt.figure(title, figsize=(20, 14))
        self.n_plots = len(data)
        self.axs = [self.fig.add_subplot(1, self.n_plots, i) for i in range(self.n_plots)]
        self.index = -1
        self.data = data
        self.styles = styles
        self.additional_information_callback = additional_information_callback

    def show(self):
        pass

    def plot_previous(self):
        if self.index > 0:
            self.index -= 1
            self.plot()

    def plot_next(self, data=None):
        if self.index < len(self.data[0]) - 1:
            self.index += 1
            self.plot()

    def plot(self):
        for k in range(self.n_plots):
            style = self.styles[k]
            if style == AttentionPlot.MATRIX_STYLE:
                self.plot_matrix(k)
            elif style == AttentionPlot.VECTOR_STYLE:
                self.plot_vector(k)

    def plot_vector(self, k, sequence=None, attention=None):
        ax = self.axs[k]
        if sequence and attention:
            self.data[k].append((sequence, attention))
            self.index = len(self.data[k]) - 1

        if 0 <= self.index < len(self.data[k]):
            d = self.data[k][self.index]
            sequence = d[0]
            attention = d[1]

            ax.clear()
            N = len(sequence)
            barlist = ax.barh(numpy.arange(N), attention[::-1])
            ax.set_xticks(numpy.arange(11.) / 11)
            ax.set_yticks(numpy.arange(N) + 2. / 5.)

            ax.set_yticklabels(sequence[::-1])
            # ax.set_xticklabels(words, rotation="vertical")
            # lefts = [item.get_x() for item in barlist]
            # rights = [item.get_x() + item.get_width() for item in barlist]
            # plt.xlim([min(lefts), max(rights)])

            if self.additional_information_callback:
                self.additional_information_callback(plt, ax, self.data[k][self.index])

            ax.grid()
            plt.draw()

            plt.ion()
            plt.show()

    def plot_matrix(self, k, sequence1=None, sequence2=None, attention=None):
        ax = self.axs[k]
        if sequence1 and sequence2 and attention:
            self.data[k].append((sequence1, sequence2, attention))
            self.index = len(self.data[k]) - 1

        if 0 <= self.index < len(self.data[k]):
            d = self.data[k][self.index]
            sequence1 = d[0]
            sequence2 = d[1]
            attention = d[2]

            # ax = self.ax
            ax.clear()

            ax.imshow(attention, interpolation='nearest', cmap=plt.cm.gray)
            ax.set_xticks(range(len(sequence2)))
            ax.set_yticks(range(len(sequence1)))

            ax.set_xticklabels(sequence2, rotation=45)
            ax.set_yticklabels(sequence1)

            if self.additional_information_callback:
                self.additional_information_callback(plt, ax, self.data[k][self.index])

            plt.draw()

            plt.ion()
            plt.show()

    def close(self):
        plt.close()

    def start(self):
        self.show()
        self.plot_next()
        while True:
            input_data = input("Press 'a' for previous, 'd' for next continue, or 'x' to exit:\n")
            if input_data == "a":
                self.plot_previous()
            if input_data == "d":
                self.plot_next()
            if input_data == "q" or input_data == "x":
                break
        self.close()


class IterablePlotter:
    class DataSample:
        def __init__(self):
            pass

    def __init__(self, title, data, plot_functions):
        # self.root = tk.Tk()
        # self.root.wm_title("Embedding in TK")
        pylab.ion()
        self.fig = pylab.figure(title, figsize=(17, 13))
        self.n_plots = len(plot_functions)
        self.axs = [self.fig.add_subplot(1, self.n_plots, i) for i in range(1, self.n_plots + 1)]
        self.index = -1
        self.data = data
        self.plot_functions = plot_functions

    def show(self):
        pass

    def plot_previous(self):
        if self.index > 0:
            self.index -= 1
            self.plot()

    def plot_next(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.plot()

    def plot(self):
        for k in range(self.n_plots):
            ax = self.axs[k]
            d = self.data[self.index]
            ax.clear()
            self.plot_functions[k](ax, d)
            ax.grid()

        self.fig.canvas.draw()
        pylab.draw()
        pylab.show()

    def close(self):
        pylab.close()

    def start(self):
        self.show()
        self.plot_next()
        while True:
            input_data = six.moves.input("Press 'a' for previous, 'd' for next continue, or 'x' to exit:\n")
            if input_data == "a":
                self.plot_previous()
            if input_data == "d":
                self.plot_next()
            if input_data == "q" or input_data == "x":
                break
        self.close()


class BatchIterator:
    def __init__(self, iterable, batch_size=1, vectorizer=None, batch_vectorizer=None, return_tuple=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.batch_vectorizer = batch_vectorizer

        self.n_batches = (len(iterable) + 1) / batch_size
        self.iterable_size = len(iterable)
        self.return_tuple = return_tuple

    # print "init iterator"

    def __iter__(self):
        for ndx in range(0, self.iterable_size, self.batch_size):
            elements = self.iterable[ndx:min(ndx + self.batch_size, self.iterable_size)]
            if self.vectorizer:
                batch = numpy.array(functools.reduce(lambda seq, elem: seq + self.vectorizer(elem), elements, []))
            else:
                batch = elements

            if self.batch_vectorizer:
                batch = self.batch_vectorizer(batch)

            if self.return_tuple:
                yield (elements, batch)
            else:
                yield batch


class BatchGenerator:
    def __init__(self, generator, batch_size=1, vectorizer=None, batch_vectorizer=None, return_tuple=False):
        self.generator = generator
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.batch_vectorizer = batch_vectorizer
        self.return_tuple = return_tuple

    def __iter__(self):
        elements = []
        for elem in self.generator:
            elements.append(elem)
            if len(elements) == self.batch_size:
                yield self.process_batch(elements)
                elements = []
        if len(elements) > 0:
            yield self.process_batch(elements)

    def process_batch(self, elements):
        if self.vectorizer:
            batch = numpy.array(functools.reduce(lambda seq, elem: seq + self.vectorizer(elem), elements, []))
        else:
            batch = elements

        if self.batch_vectorizer:
            batch = self.batch_vectorizer(batch)

        if self.return_tuple:
            return (elements, batch)
        else:
            return batch


class MultiBatchIterator:
    def __init__(self, iterable, batch_size=1, vectorizer=None, batch_vectorizer=None, return_tuple=False,
                 allow_multiple_vectors_per_element=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.batches_vectorizer = batch_vectorizer

        self.n_batches = (len(iterable) + 1) / batch_size
        self.n_data = len(iterable)
        self.return_tuple = return_tuple
        self.allow_multiple_vectors_per_element = allow_multiple_vectors_per_element
        self.batch_index = 0
        self.data_index = 0

    def __iter__(self):
        batches = []
        elements = []
        self.data_index = 0
        self.batch_index = 0
        for element in self.iterable:
            self.data_index += 1
            # print "--- New Element"
            if self.vectorizer:
                vectorized_batch_parts = list(self.vectorizer(element))
            # print "batches", get_padding_shape(batches)
            # print "vectorized", get_padding_shape(vectorized_batch_parts)
            else:
                vectorized_batch_parts = element
            if len(batches) == 0:
                batches = [[] for b_i in range(len(vectorized_batch_parts))]

            for i, (batch, batch_part) in enumerate(zip(batches, vectorized_batch_parts)):
                # print i, "batch", get_padding_shape(batch)
                # print i, "batch_part", get_padding_shape(batch_part)
                if self.allow_multiple_vectors_per_element:
                    # print len(batch), len(batch_part)
                    batch += batch_part
                # print len(batch)
                else:
                    # print "????"
                    batch.append(batch_part)

            # print "--- batches", get_padding_shape(batches)

            elements.append(element)

            if len(batches[0]) >= self.batch_size:
                self.batch_index += 1
                yield self.__vectorize_batch(batches, elements)
                batches = []
                elements = []

        if len(batches) > 0 and len(batches[0]) >= 0:
            yield self.__vectorize_batch(batches, elements)
            batches = []
            elements = []

    def __vectorize_batch(self, batches, elements):
        if self.batches_vectorizer:
            batches = self.batches_vectorizer(batches)
        # print "__vectorize_batch", get_padding_shape(batches)
        if self.return_tuple:
            return (elements, batches)
        else:
            return batches


class NamedMultiBatchIterator:
    DATA_BATCH_NAME = "raw_data"

    def __init__(self, iterable, batch_size=1, vectorizer=None, batch_vectorizer=None):
        self.iterable = iterable
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.batches_vectorizer = batch_vectorizer

        self.n_batches = (len(iterable) + 1) / batch_size
        self.n_data = len(iterable)
        self.batch_index = 0
        self.data_index = 0

    def __iter__(self):
        batches = None
        self.data_index = 0
        self.batch_index = 0
        for element in self.iterable:
            self.data_index += 1
            if self.vectorizer:
                vectorized_batch_parts = self.vectorizer(element)
            else:
                vectorized_batch_parts = element

            if batches is None:
                # create empty dict
                batches = BetterDict((name, []) for name, part in vectorized_batch_parts.items())
                batches[NamedMultiBatchIterator.DATA_BATCH_NAME] = []

            # append new element to each batch
            for name in vectorized_batch_parts:
                batch = batches[name]
                batch_part = vectorized_batch_parts[name]
                batch.append(batch_part)

            batches[NamedMultiBatchIterator.DATA_BATCH_NAME].append(element)

            current_batch_size = len(batches[NamedMultiBatchIterator.DATA_BATCH_NAME])
            if current_batch_size >= self.batch_size:
                self.batch_index += 1
                yield self.__vectorize_batch(batches)
                batches = None

        if batches is not None:
            yield self.__vectorize_batch(batches)
            batches = None

    def __vectorize_batch(self, batches):
        if self.batches_vectorizer:
            vectorized_batches = self.batches_vectorizer(batches)
            vectorized_batches[NamedMultiBatchIterator.DATA_BATCH_NAME] = batches[
                NamedMultiBatchIterator.DATA_BATCH_NAME]
            batches = BetterDict(vectorized_batches)
        return batches


def print_batch_shapes(batches):
    for name, b in batches.items():
        if hasattr(b, "shape"):
            shape = b.shape
        else:
            shape = len(b)

        print("'{}': {}".format(name, shape))


class IteratorSampler:
    def __init__(self, batch_iterators):
        self.batch_iterators = batch_iterators

    def __iter__(self):
        iterators = [iter(bi) for bi in self.batch_iterators]
        while True:
            n_non_empty_iterators = functools.reduce(lambda acc, it: acc + (it.data_index < it.n_data),
                                                     self.batch_iterators, 0)
            if n_non_empty_iterators > 0:
                processed = numpy.array(
                    [1 - float(it.data_index) / it.n_data if it.n_data > 0 else 0 for it in self.batch_iterators])
                prob = processed / numpy.sum(processed)
                it_order = numpy.random.choice(range(len(iterators)), p=prob, size=n_non_empty_iterators, replace=False)
                for sampled_index in it_order:
                    it = iterators[sampled_index]
                    try:
                        batches = it.next()
                        yield sampled_index, batches
                        break
                    except StopIteration:
                        pass
            else:
                break


class Writer(object):
    def __init__(self, dirpath, max_lines=100, max_files_per_directory=1000, compress=True, verbose=False):
        self.dirpath = dirpath
        self.max_lines = max_lines
        self.compress = compress
        self.max_files_per_directory = max_files_per_directory
        self.output_file = None
        self.current_dir_count = 0
        self.current_file_count = 0
        self.n_lines = 0
        self.verbose = verbose
        self.index_file = None

    def write(self, text, id=None):
        if self.n_lines >= self.max_lines:
            self.close()
        if self.output_file is None:
            self._next()

        self.output_file.write(text + "\n")
        if id is not None:
            self._write_index(id)

        self.n_lines += 1

    def _write_index(self, id):
        if self.index_file is None:
            self.index_file = open(os.path.join(self.dirpath, "index.tsv"))
        self.index_file.write(id + "\t" + "\n")

    def _next(self):
        if self.current_file_count >= self.max_files_per_directory:
            self.current_dir_count += 1
            self.current_file_count = 0
            if self.verbose:
                print("Open new directory: ", self.current_dir_count)

        dirpath = os.path.join(self.dirpath, str(self.current_dir_count))
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        self.current_file_count += 1
        self.n_lines = 0

        # if self.verbose:
        #     print "Open new file: ", self.current_file_count

        suffix = ".json"
        if self.compress:
            suffix += ".gz"

        filepath = os.path.join(dirpath, str(self.current_file_count) + suffix)

        if self.compress:
            self.output_file = gzip.open(filepath, "w")
        else:
            self.output_file = open(filepath, "w")

    def close(self):
        if self.output_file:
            self.output_file.close()
        if self.index_file:
            self.index_file.close()
        self.output_file = None
        self.index_file = None


class ExperimentFiles(object):
    def __init__(self, experiment_name, conf, base_dirpath):
        self.experiment_name = experiment_name
        self.conf = conf
        self.base_dirpath = base_dirpath
        self.experiment_id = self.experiment_name + "_" + get_timestamp()
        self.conf.experiment_id = self.experiment_id

        self._experiment_dir = os.path.join(self.base_dirpath, self.experiment_id)
        self._models_dir = os.path.join(self.base_dirpath, "models")
        self._output_dir = os.path.join(self.base_dirpath, "output")
        self._log_dir = os.path.join(self.base_dirpath, "log")
        self._results_dir = os.path.join(self.base_dirpath, "results")

    def _init_dir(self, dirpath):
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

    @property
    def experiment_dir(self):
        self._init_dir(self._experiment_dir)
        return self._experiment_dir

    @property
    def models_dir(self):
        self._init_dir(self._models_dir)
        return self._models_dir

    @property
    def output_dir(self):
        self._init_dir(self._output_dir)
        return self._output_dir

    @property
    def log_dir(self):
        self._init_dir(self._log_dir)
        return self._log_dir

    @property
    def results_dir(self):
        self._init_dir(self._results_dir)
        return self._results_dir


class ModelStorage(object):
    def __init__(self, model_dirpath, model_weight_filename="best_model.weights.h5"):
        self.best_score = None
        self.best_epoch = None
        self.best_model_filepath = os.path.join(model_dirpath, model_weight_filename)

    def save_best(self, model, score, epoch):
        if self.best_score is None or score > self.best_score:
            print("Save new best model at epoch {} with score {}.".format(epoch, score))
            self.best_score = score
            self.best_epoch = epoch
            model.save_weights(self.best_model_filepath)
            return True

        return False


# def _get_colors(num_colors):
# 	colors = []
# 	for i in np.arange(0., 360., 360. / num_colors):
# 		hue = i / 360.
# 		lightness = (50 + np.random.rand() * 10) / 100.
# 		saturation = (90 + np.random.rand() * 10) / 100.
# 		colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
# 	return colors


def pad_to_shape(X, to_shape, padding_position, value, mask=False):
    to_shape = list(to_shape)
    X_padded = []
    X_padding_mask = []
    if len(to_shape) > 1:
        for x in X:
            x_padded, padding_mask = pad_to_shape(x, to_shape[1:], padding_position, value, mask=True)
            X_padded.append(x_padded)
            X_padding_mask.append(padding_mask)
    else:
        X_padded = X
        X_padding_mask = numpy.ones((len(X)))

    X_padded = numpy.array(X_padded)
    X_padding_mask = numpy.array(X_padding_mask)

    padding_value = numpy.array(value)
    pad_shape = [to_shape[0] - len(X)] + to_shape[1:] + [1] * padding_value.ndim
    X_pad = numpy.ones(pad_shape) * padding_value
    # print X_padded.shape, pad_shape, X_pad.shape
    X_pad_mask = numpy.zeros_like(X_pad)
    # print X_padding_mask.shape, X_pad_mask.shape
    if padding_position == "pre":
        X_padded = numpy.concatenate((X_pad, X_padded), axis=0)

        X_padding_mask = numpy.concatenate((X_pad_mask, X_padding_mask), axis=0)
    elif padding_position == "post":
        X_padded = numpy.concatenate((X_padded, X_pad), axis=0)
        X_padding_mask = numpy.concatenate((X_padding_mask, X_pad_mask), axis=0)

    if mask:
        return X_padded, X_padding_mask
    else:
        return X_padded


def pad_to_shape_no_mask(X, to_shape, padding_position, value):
    to_shape = list(to_shape)
    X_padded = []
    if len(to_shape) > 1:
        for x in X:
            x_padded = pad_to_shape_no_mask(x, to_shape[1:], padding_position, value)
            X_padded.append(x_padded)
    else:
        X_padded = X

    X_padded = numpy.array(X_padded)

    if isinstance(value, string_types):
        padding_value = [value]
        pad_shape = [to_shape[0] - len(X)] + to_shape[1:]
    else:
        padding_value = numpy.array(value)
        pad_shape = [to_shape[0] - len(X)] + to_shape[1:] + [1] * padding_value.ndim

    X_pad = numpy.ones(pad_shape) * padding_value
    if padding_position == "pre":
        X_padded = numpy.concatenate((X_pad, X_padded), axis=0)

    elif padding_position == "post":
        X_padded = numpy.concatenate((X_padded, X_pad), axis=0)

    return X_padded


def pad(X, padding_position, value, mask=False):
    shape = get_padding_shape(X)
    # print "RAW PAD SHAPE:", shape
    value_shape = get_padding_shape(value)

    n_dim_value = len(value_shape)
    if n_dim_value > 0:
        shape = shape[:-n_dim_value]  # print "Value shape:", value_shape, n_dim_value  # print "PAD SHAPE:", shape

    if mask:
        return pad_to_shape(X, shape, padding_position, value, True)
    else:
        return pad_to_shape_no_mask(X, shape, padding_position, value)


def get_padding_shape(X):
    if isinstance(X, string_types):
        return []

    try:
        iter(X)
    except TypeError as e:
        return []

    # if not isinstance(X, list) and not isinstance(X, tuple):
    # 	return None
    padding_shape_x = []
    for x in X:
        padding_shape_x_new = get_padding_shape(x)

        if len(padding_shape_x_new) == 0:
            return [len(X)]
        else:
            if len(padding_shape_x) == 0:
                padding_shape_x = padding_shape_x_new
            else:
                padding_shape_x = list(
                    map(max, itertools.zip_longest(padding_shape_x, padding_shape_x_new, fillvalue=0)))
    return [len(X)] + padding_shape_x


def remove_padding(sequence, expected_length, padding_position):
    if padding_position == "pre":
        return sequence[len(sequence) - expected_length:]
    elif padding_position == "post":
        return sequence[:expected_length]


def get_timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


def save_model(model, model_name, model_dir):
    try:
        with open(os.path.join(model_dir, model_name + ".keras.model.json"), "w") as f:
            json_string = model.to_json()
            f.write(json_string)
    except TypeError as e:
        print(e)

    model.save_weights(model_dir + model_name + '.keras.weights.h5', overwrite=True)


def load_model(model_name, model_dir):
    from keras.models import model_from_json

    with open(os.path.join(model_dir, model_name + ".keras.model.json")) as f:
        json_string = f.read()
        model = model_from_json(json_string)
        model.load_weights(model_dir + model_name + '.keras.weights.h5')
        return model


# def save_data(dataset, dataset_name, data_dir):
#     try:
#         import sys
#         sys.setrecursionlimit(10000)
#         dataset_file = open(data_dir + dataset_name + ".cpickle", "w")
#         cPickle.dump(dataset, dataset_file)
#         dataset_file.close()
#         print("%s cPickled to file" % dataset_name)
#     except RuntimeError as e:
#         print e
#     except IOError as e:
#         print e
#
#
# def load_data(dataset_name, data_dir):
#     import sys
#     sys.setrecursionlimit(10000)
#     print("load model from file %s" % (data_dir + dataset_name + ".cpickle"))
#     dataset_file = open(data_dir + dataset_name + ".cpickle")
#     dataset = cPickle.load(dataset_file)
#     dataset_file.close()
#     print("%s loaded from file" % dataset_name)
#     return dataset
#
#
# def save_corpus(corpus_wrapper, corpus_name, corpus_dir):
#     try:
#         import sys
#         sys.setrecursionlimit(10000)
#         corpus_file = open(corpus_dir + corpus_name + ".corpus.cpickle", "w")
#         cPickle.dump(corpus_wrapper, corpus_file)
#         corpus_file.close()
#         print("%s cPickled to file" % corpus_name)
#     except RuntimeError as e:
#         print e
#     except IOError as e:
#         print e
#
#
# def load_corpus(corpus_name, corpus_dir):
#     import sys
#     sys.setrecursionlimit(10000)
#     print("load model from file %s" % (corpus_dir + corpus_name + ".corpus.cpickle"))
#     corpus_file = open(corpus_dir + corpus_name + ".corpus.cpickle")
#     corpus_wrapper = cPickle.load(corpus_file)
#     corpus_file.close()
#     print("%s loaded from file" % corpus_name)
#     return corpus_wrapper


def log(file, text, file_only=False):
    if file:
        file.write(text + "\n")
    if not file_only:
        print(text)


def _remove_vowls(word):
    return re.sub("(?<=[^_])[aeiou]", "", word)


def dict2name(d):
    name = ""
    separator = ""
    items = sorted(d.items())
    for k, v in items:
        name += separator + "%s=%s" % (_remove_vowls(k), v)
        separator = "_"
    return name


def seconds_to_readable_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def write_vocab(vocab, filepath):
    with open(filepath, "w") as f:
        for v, c in vocab.items():
            f.write(v + " " + str(c))
            f.write("\n")


def load_vocab(filepath):
    with open(filepath) as f:
        vocab = Counter()
        for l in f:
            v, c = l.split()
            vocab[v] = int(c)
    return vocab


def write_object(obj, filepath):
    with open(filepath, "w") as f:
        f.write(obj.__str__())
        f.close()


def load_object(filepath):
    with open(filepath) as f:
        l = f.readline()
        obj = eval(l)
    return obj


def write_iterable(it, filepath, to_str=None, encoding="utf-8"):
    with io.open(filepath, "w", encoding=encoding) as f:
        for x in it:
            if to_str:
                x = to_str(x)
            line = x + u"\n"
            f.write(line)


def write_index2word_dict(index2word, filepath):
    with io.open(filepath, "w", encoding="utf-8") as f:
        for idx, word in index2word.items():
            f.write(u"%s %s\n" % (idx, word))


def load_as_list(filepath, to_object=None, filter=None, encoding="utf-8"):
    s = list()
    with io.open(filepath, encoding=encoding) as f:
        for line in f:
            line = line.rstrip()
            if filter is None or filter(line):
                if to_object:
                    obj = to_object(line)
                else:
                    obj = line
                s.append(obj)
    return s


def load_as_set(filepath, to_object=None, filter=None, encoding="utf-8"):
    s = set()
    with io.open(filepath, encoding=encoding) as f:
        for line in f:
            element = line.rstrip()
            if filter is None or filter(element):
                if to_object:
                    element = to_object(element)
                s.add(element)
    return s


def load_as_dict(filepath, sep=" ", to_key_value=None):
    s = dict()
    with io.open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if to_key_value:
                key, value = to_key_value(line)
            else:
                parts = line.split(sep)
                if len(parts) != 2:
                    print("ERROR in line %s: %s" % (i, line))
                key, value = parts

            s[key] = value
    return s


def load_as_counter(filepath, sep=" ", to_key_value=None, skip_errors=False):
    s = Counter()
    with io.open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            if to_key_value:
                key, value = to_key_value(line)
            else:
                parts = line.split(sep)
                if len(parts) != 2:
                    print("ERROR in line %s: %s" % (i, line))
                    print("ERROR %s parts: %s" % (len(parts), parts))
                    if skip_errors:
                        continue
                key, value = parts

            s[key] = int(value)
    return s


def load_mapping(filepath):
    with open(filepath) as f:
        l = f.readline()
        index2token = eval(l)
        token2index = dict([(t, i) for i, t in index2token.items()])

    return index2token, token2index


def patient_log(message, i, patience, f=None):
    if i > 0 and i % patience == 0:
        log(f, "{} @ {}".format(message, get_timestamp()))


def colorize(text, flag):
    return (Fore.GREEN if flag else Fore.RED) + str(text) + Fore.RESET


def plot_simple_word_attention_function(ax, data):
    words, attention = data
    N = len(words)
    barlist = ax.bar(numpy.arange(N), attention)
    ax.set_xticks(numpy.arange(N) + 2. / 5.)
    ax.set_yticks(numpy.arange(11.) / 11)

    ax.set_xticklabels(words, rotation=30)


def plot_k_word_attention_function(ax, data):
    words, attentions = data
    colors = ["r", "b", "g", "y", "c", "m", "k", "#990000", "#009900", "#000099", "#ff9922"]
    N = len(words)
    width = 0.8 / len(attentions)
    offsets = numpy.array(range(len(attentions))) - float(len(attentions)) / 2
    for i, ((name, attention), offset) in enumerate(zip(attentions, offsets)):
        barlist = ax.bar(numpy.arange(N) - width * offset, attention, width=width, align='center', label=name,
                         color=colors[i])

    ax.set_xticks(numpy.arange(N) + 2. / 5.)
    ax.set_yticks(numpy.arange(11.) / 11)

    ax.set_xticklabels(words, rotation=30)
    pylab.legend(loc="upper left")


def align_string_lists(lists, padding_character=" ", alignment_position="start"):
    aligned_lists = []

    for elements in zip(*lists):
        lengths = [len(e) for e in elements]
        max_len = max(lengths)
        if alignment_position == "start":
            elements = [e.ljust(max_len, padding_character) for e in elements]
        elif alignment_position == "end":
            elements = [e.rjust(max_len, padding_character) for e in elements]
        aligned_lists.append(elements)

    aligned_lists = zip(*aligned_lists)
    return aligned_lists


def get_sampled_configuration(base_conf: Configuration, param_distribution: dict, n_iter: int, seed: int) -> List[
    Configuration]:
    random_state = RandomState(seed)

    n_confs = 1
    for params in param_distribution.values():
        n_confs *= len(params)

    if n_iter > n_confs:
        print("WARNING: Parameter grid only contains {} distinct configurations.".format(n_confs))
        n_iter = n_confs

    sampler = ParameterSampler(param_distribution, n_iter, random_state)
    sampled_params_list = [params for params in sampler]

    confs = []
    for params in sampled_params_list:
        conf = Configuration(base_conf)
        for k, v in params.items():
            conf[k] = v
        confs.append(conf)

    return confs


def insert_dependent_params(conf: Configuration, param_maps: Sequence[Tuple[str, str, dict]],
                            ignore_missing=False) -> Configuration:
    mapped_conf = Configuration(conf)
    for source_key, target_key, param_map in param_maps:
        source_value = mapped_conf[source_key]
        if source_value in param_map:
            mapped_conf[target_key] = param_map[source_value]
        elif None in param_map:
            mapped_conf[target_key] = param_map[None]
        else:
            if not ignore_missing:
                raise ValueError("Parameter map for {} does not contain entry for source value: '{}'".format(
                    (source_key, target_key), source_value))

    return mapped_conf


def get_named_keras_model_outputs(model, outputs):
    if not isinstance(outputs, list):
        outputs = [outputs]
    return dict(zip(model.output_names, outputs))
