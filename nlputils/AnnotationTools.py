from colorama import Fore
import numpy
from bisect import bisect_left

import DataTools

__author__ = 'sjebbara'


class IOBScheme:
    index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
    size = len(index2tag)
    name = "IOB"

    @staticmethod
    def spans2tags(length, spans):
        sorted(spans, key=lambda s: s[0])
        bio_sequence = ["O"] * length
        for s, e in spans:
            if bio_sequence[s - 1] == "B" or bio_sequence[s - 1] == "I":
                # if previous token is annotated (by different annotation)
                # == if last is I or B start with B
                bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
            else:
                # if last is O start with I
                bio_sequence[s:e] = ["I"] * (e - s)

        return bio_sequence

    @staticmethod
    def tags2spans(tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "B":
                if start is not None:
                    spans.append((start, i))
                start = i
            elif t == "I":
                if start is None:
                    start = i
            elif t == "O":
                if start is not None:
                    spans.append((start, i))
                start = None

        if start:
            spans.append((start, len(tags)))
        return spans

    @staticmethod
    def spans2encoding(length, spans):
        return IOBScheme.tags2encoding(IOBScheme.spans2tags(length, spans))

    @staticmethod
    def tags2encoding(tags):
        return tag_sequence2encoding(tags, IOBScheme.tag2index)

    @staticmethod
    def encoding2tags(encoding):
        return encoding2tag_sequence(encoding, IOBScheme.index2tag)

    @staticmethod
    def encoding2spans(encoding):
        return IOBScheme.tags2spans(encoding2tag_sequence(encoding, IOBScheme.index2tag))

    @staticmethod
    def visualize_encoding(elements, bio, onehot=True, spacer=u""):
        b = Fore.GREEN  # beginning
        i = Fore.BLUE  # inside
        o = Fore.BLACK  # outside
        BI = [b, i, o]
        if onehot:
            bio_indices = numpy.argmax(bio, axis=1)
        else:
            bio_indices = bio
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bio_indices)))
        return colored_text

    @staticmethod
    def visualize_tags(elements, tags, spacer=u""):
        cm = dict()
        cm["B"] = Fore.GREEN  # beginning
        cm["I"] = Fore.BLUE  # inside
        cm["O"] = ""  # outside
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda (e, t): cm[t] + e + Fore.RESET, zip(elements, tags)))
        return colored_text

    @staticmethod
    def visualize_spans(elements, spans, spacer=u""):
        tags = IOBScheme.spans2tags(len(elements), spans)
        return IOBScheme.visualize_tags(elements, tags, spacer)

    def __str__(self):
        return IOBScheme.name

    def __repr__(self):
        return IOBScheme.name


class IOB2Scheme:
    index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
    size = len(index2tag)
    name = "IOB2"

    @staticmethod
    def spans2tags(length, spans):
        sorted(spans, key=lambda s: s[0])
        bio_sequence = ["O"] * length
        for s, e in spans:
            bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
        return bio_sequence

    @staticmethod
    def tags2spans(tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "B":
                if start is not None:
                    spans.append((start, i))
                start = i
            elif t == "I":
                if start is None:
                    start = i
            elif t == "O":
                if start is not None:
                    spans.append((start, i))
                start = None

        if start:
            spans.append((start, len(tags)))
        return spans

    @staticmethod
    def spans2encoding(length, spans):
        return IOB2Scheme.tags2encoding(IOB2Scheme.spans2tags(length, spans))

    @staticmethod
    def tags2encoding(tags):
        return tag_sequence2encoding(tags, IOB2Scheme.tag2index)

    @staticmethod
    def encoding2tags(encoding):
        return encoding2tag_sequence(encoding, IOB2Scheme.index2tag)

    @staticmethod
    def encoding2spans(encoding):
        return IOB2Scheme.tags2spans(encoding2tag_sequence(encoding, IOB2Scheme.index2tag))

    @staticmethod
    def visualize_encoding(elements, bio, onehot=True, spacer=u""):
        b = Fore.GREEN  # beginning
        i = Fore.BLUE  # inside
        o = Fore.BLACK  # outside
        BI = [b, i, o]
        if onehot:
            bio_indices = numpy.argmax(bio, axis=1)
        else:
            bio_indices = bio
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bio_indices)))
        return colored_text

    def __str__(self):
        return IOBScheme.name

    def __repr__(self):
        return IOBScheme.name


# class IOEScheme:
#     index2tag, tag2index = DataTools.get_mappings(["E", "I", "O"])
#     size = len(index2tag)
#     name = "IOE"
#
#     @staticmethod
#     def annotations2encoding(n, annotations):
#         return IOEScheme.spans2encoding(n, annotations_to_spans(annotations))
#
#     @staticmethod
#     def spans2encoding(length, spans):
#         return tag_sequence2encoding(spans2iob_sequence(length, spans), IOEScheme.tag2index)
#
#     @staticmethod
#     def encoding2spans(tag_probabilities):
#         return iob_sequence2spans(encoding2tag_sequence(tag_probabilities, IOEScheme.index2tag))
#
#     @staticmethod
#     def visualize_encoding(elements, bio, onehot=True, spacer=""):
#         b = Fore.GREEN  # beginning
#         i = Fore.BLUE  # inside
#         o = Fore.BLACK  # outside
#         BI = [b, i, o]
#         if onehot:
#             bio_indices = numpy.argmax(bio, axis=1)
#         else:
#             bio_indices = bio
#         # outside should not be colored at all
#         colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bio_indices)))
#         return colored_text
#
#     def __str__(self):
#         return self.name
#
#     def __repr__(self):
#         return self.name


# class IOB2Scheme:
#     index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
#     size = len(index2tag)
#
#     def __init__(self):
#         self.name = "IOB2"
#         self.annotations2encoding = annotations_to_iob2
#         self.spans2encoding = spans2iob2
#         self.encoding2spans = iob22spans
#         self.visualize_encoding = visualize_iob2
#         self.spans2tags = spans2iob2_sequence
#
#     def __str__(self):
#         return self.name
#
#     def __repr__(self):
#         return self.name


# class IOBESScheme:
#     index2tag, tag2index = DataTools.get_mappings(["S", "B", "I", "E", "O"])
#     size = len(index2tag)
#
#     def __init__(self):
#         self.name = "IOBES"
#         self.annotations2encoding = annotations_to_iobes
#         self.spans2encoding = spans2iobes
#         self.encoding2spans = iobes2spans
#         self.visualize_encoding = visualize_iobes
#         self.spans2tags = spans2iobes_sequence
#
#     def __str__(self):
#         return self.name
#
#     def __repr__(self):
#         return self.name


def binary_search(a, x, lo=0, hi=None):  # can't use a to specify default for hi
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)
    pos = bisect_left(a, x, lo, hi)  # find insertion position
    return (pos if pos != hi and a[pos] == x else -1)  # don't walk off the end


def find_matching_spans(spans, start, end):
    start_span_index = None
    end_span_index = None
    for i, (s, e) in enumerate(spans):
        if start_span_index is None and s <= start < e:
            start_span_index = i
        elif start_span_index is None and i > 0 and spans[i - 1][1] <= start <= s:
            start_span_index = i

        if end_span_index is None and s < end <= e:
            end_span_index = i + 1
        elif end_span_index is None and i < len(spans) - 1 and e < end <= spans[i + 1][0]:
            end_span_index = i + 1

    if end_span_index is None:
        end_span_index = len(spans)
    return start_span_index, end_span_index


def spans2tags(length, spans, annotation_scheme="IOB"):
    if annotation_scheme == "IOB":
        return spans2iob(length, spans)
    elif annotation_scheme == "IOB2":
        return spans2iob2(length, spans)
        # elif annotation_scheme == "IOBES":
        #     return spans2iobes(length, spans)


####### IOB ########
def spans2iob(length, spans):
    return tag_sequence2encoding(spans2iob_sequence(length, spans), IOBScheme.tag2index)


def spans2iob_sequence(length, spans):
    spans.sort(key=lambda s: s[0])
    bio_sequence = ["O"] * length
    for s, e in spans:
        if bio_sequence[s - 1] == "B" or bio_sequence[s - 1] == "I":
            bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
        else:
            bio_sequence[s:e] = ["I"] * (e - s)

    return bio_sequence


def visualize_iob(elements, bio, onehot=True, spacer=u""):
    b = Fore.GREEN  # beginning
    i = Fore.BLUE  # inside
    o = Fore.BLACK  # outside
    BI = [b, i, o]
    if onehot:
        bio_indices = numpy.argmax(bio, axis=1)
    else:
        bio_indices = bio
    # outside should not be colored at all
    colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bio_indices)))
    return colored_text


def annotations_to_iob(n, annotations):
    return spans2iob(n, annotations_to_spans(annotations))


def iob_sequence2spans(tag_sequence):
    spans = []
    start = None
    for i, t in enumerate(tag_sequence):
        if t == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "I":
            if start is None:
                start = i
        elif t == "O":
            if start is not None:
                spans.append((start, i))
            start = None

    if start:
        spans.append((start, len(tag_sequence)))
    return spans


def iob2spans(Y_pred):
    return iob_sequence2spans(encoding2tag_sequence(Y_pred, IOBScheme.index2tag))


####### IOB 2 ########
def spans2iob2(length, spans):
    return tag_sequence2encoding(spans2iob2_sequence(length, spans), IOB2Scheme.tag2index)


def spans2iob2_sequence(length, spans):
    spans.sort(key=lambda s: s[0])
    bio_sequence = ["O"] * length
    for s, e in spans:
        bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
    return bio_sequence


def visualize_iob2(elements, bio, onehot=True, spacer=u""):
    b = Fore.GREEN  # beginning
    i = Fore.BLUE  # inside
    o = Fore.BLACK  # outside
    BI = [b, i, o]
    if onehot:
        bio_indices = numpy.argmax(bio, axis=1)
    else:
        bio_indices = bio
    # outside should not be colored at all
    colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bio_indices)))
    return colored_text


def annotations_to_iob2(n, annotations):
    return spans2iob2(n, annotations_to_spans(annotations))


def iob2_sequence2spans(tag_sequence):
    spans = []
    start = None
    for i, t in enumerate(tag_sequence):
        if t == "B":
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == "I":
            if start is None:
                start = i
        elif t == "O":
            if start is not None:
                spans.append((start, i))
            start = None

    if start:
        spans.append((start, len(tag_sequence)))
    return spans


def iob22spans(Y_pred):
    return iob2_sequence2spans(encoding2tag_sequence(Y_pred, IOB2Scheme.index2tag))


# def spans2iobes(length, spans):
#     return tag_sequence2encoding(spans2iobes_sequence(length, spans), IOBESScheme.tag2index)
#
# def spans2iobes_sequence(length, spans):
#     spans.sort(key=lambda s: s[0])
#     bio_sequence = ["O"] * length
#     for s, e in spans:
#         if e - s > 1:
#             bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 2) + ["E"]
#         else:
#             bio_sequence[s:e] = ["S"]
#
#     return bio_sequence
#
# def visualize_iobes(elements, bilou, onehot=True, spacer=u""):
#     b = Fore.GREEN  # beginning
#     i = Fore.BLUE  # inside
#     l = Fore.RED  # last
#     o = Fore.BLACK  # outside
#     u = Fore.MAGENTA  # (single) unit
#     BI = [b, i, l, o, u]
#     if onehot:
#         bilou_indices = numpy.argmax(bilou, axis=1)
#     else:
#         bilou_indices = bilou
#     colored_text = spacer.join(map(lambda (e, bi): BI[bi] + e + Fore.RESET, zip(elements, bilou_indices)))
#     return colored_text
#
# def annotations_to_iobes(n, annotations):
#     return spans2iobes(n, annotations_to_spans(annotations))

def tag_sequence2encoding(tag_sequence, tag2index):
    n_tags = len(tag2index)
    n_elements = len(tag_sequence)
    encoding = numpy.zeros((n_elements, n_tags))
    for i, t in enumerate(tag_sequence):
        encoding[i, tag2index[t]] = 1

    return encoding


def annotations_to_spans(annotations):
    return map(lambda a: (a.start, a.end), annotations)


def tags2spans(tags, annotation_scheme="IOB"):
    if annotation_scheme == "IOB":
        return iob2spans(tags)
    if annotation_scheme == "IOB2":
        return iob22spans(tags)
        # elif annotation_scheme == "IOBES":
        #     return iobes2spans(tags)


def encoding2tag_sequence(encoding, index2tag):
    indices = numpy.argmax(encoding, axis=1)
    tag_sequence = DataTools.indices2values(indices, index2tag)
    return tag_sequence


# def iobes2spans(Y_pred):
#     # TODO unfinished
#     b = 0  # beginning
#     i = 1  # inside
#     l = 2  # last
#     o = 3  # outside
#     u = 4  # (single) unit
#     Y_pred_classes = numpy.argmax(Y_pred, axis=1)
#     spans = []
#     start = None
#     for k, c in enumerate(Y_pred_classes):
#         if c == b:
#             if start is not None:
#                 spans.append((start, k))
#
#             start = k
#         elif c == i:
#             if start is None:
#                 start = k
#         elif c == l:
#             if start is not None:
#                 spans.append((start, k))
#                 start = None
#         elif c == o:
#             if start is not None:
#                 spans.append((start, k))
#                 start = None
#         elif c == u:
#             if start is not None:
#                 spans.append((start, k))
#                 start = None
#
#             spans.append((k, k + 1))
#
#     if start:
#         spans.append((start, len(Y_pred_classes)))
#
#     return spans


def is_inside_span(i, span):
    return span[0] <= i < span[1]


def overlap(s1, s2):
    return max(0, min(s1[1], s2[1]) - max(s1[0], s2[0]))


def size(span):
    return max(0, span[1] - span[0])


def span_distance(s1, s2):
    a = s1[0]
    b = s1[1]
    x = s2[0]
    y = s2[1]
    return max(a - y, x - b, -1)


def span_offset(s1, s2):
    a = s1[0]
    b = s1[1]
    x = s2[0]
    y = s2[1]

    o1 = x - b + 1
    o2 = y - a - 1
    if numpy.sign(o1) != numpy.sign(o2):
        return 0
    else:
        if numpy.abs(o1) < numpy.abs(o2):
            return o1
        else:
            return o2


def make_spans_match_tokens(spans, token_spans):
    matched_spans = []
    for s in spans:
        matching_tokens = []
        for t in token_spans:
            if overlap(s, t) >= float(size(t)) / 3:
                matching_tokens.append(t)
        if len(matching_tokens) > 0:
            matched_spans.append(reduce(lambda n, t: (min(t[0], s[0]), max(t[1], s[1])), matching_tokens, s))
    return matched_spans


def is_touching(prev_span, span):
    return prev_span[1] == span[0]
