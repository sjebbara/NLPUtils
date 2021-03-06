import functools
import numpy
import sys
from colorama import Fore
from bisect import bisect_left
from nlputils import DataTools

__author__ = 'sjebbara'


class AbstractTaggingScheme(object):
    colors = {"O": Fore.BLACK, "B": Fore.GREEN, "I": Fore.CYAN, "E": Fore.BLUE, "S": Fore.MAGENTA}
    index2tag, tag2index = DataTools.get_mappings(["O", "B", "I", "E", "S"])

    # size = len(index2tag)

    @classmethod
    def size(cls):
        return len(cls.index2tag)

    @classmethod
    def spans2tags(cls, length, spans):
        raise NotImplementedError()

    @classmethod
    def tags2spans(cls, tags):
        raise NotImplementedError()

    @classmethod
    def encoding2tags(cls, encoding):
        return encoding2tag_sequence(encoding, cls.index2tag)

    @classmethod
    def spans2encoding(cls, length, spans):
        return cls.tags2encoding(cls.spans2tags(length, spans))

    @classmethod
    def encoding2spans(cls, encoding):
        return cls.tags2spans(cls.encoding2tags(encoding))

    @classmethod
    def tags2encoding(cls, tag_sequence):
        encoding = numpy.zeros((len(tag_sequence), cls.size()))
        for i, tag in enumerate(tag_sequence):
            encoding[i, cls.tag2index[tag]] = 1
        return encoding

    @classmethod
    def visualize_encoding(cls, elements, bio, spacer=u""):
        colored_text = cls.visualize_tags(elements, cls.encoding2tags(bio), spacer)
        return colored_text

    @classmethod
    def visualize_tags(cls, elements, tags, spacer=u""):
        # outside should not be colored at all
        colored_text = spacer.join(
            [cls.colors[tag] + element + Fore.RESET if tag != "O" else element for element, tag in zip(elements, tags)])
        return colored_text


class SafeDecodingScheme(AbstractTaggingScheme):
    @classmethod
    def encoding2tags(cls, encoding):
        encoding = numpy.array(encoding)
        tags = []

        outside_index = cls.tag2index["O"]

        for vec in encoding:
            outside = vec[outside_index]

            if outside > 0.5:
                tags.append("O")
            else:
                tmp_vec = numpy.array(vec)
                tmp_vec[outside_index] = 0

                max_i = numpy.argmax(tmp_vec)
                max_tag = cls.index2tag[max_i]
                tags.append(max_tag)
        return tags

        # @classmethod
        # def encoding2tags(cls, encoding):
        #     encoding = numpy.array(encoding)
        #
        #     outside_index = cls.tag2index["O"]
        #     outside_scores = encoding[:, outside_index]
        #
        #     inside_scores = numpy.array(encoding)
        #     inside_scores[:, outside_index] = 0
        #     inside_scores = numpy.sum(inside_scores, axis=1)
        #
        #     binary_tags = list(inside_scores > outside_scores)
        #
        #     tags = ["I" if b else "O" for b in binary_tags]
        #     return tags


class IOE2Scheme(AbstractTaggingScheme):
    index2tag, tag2index = DataTools.get_mappings(["O", "I", "E"])

    # size = len(index2tag)

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        bio_sequence = ["O"] * length
        for s, e in spans:
            bio_sequence[s:e] = ["I"] * (e - s - 1) + ["E"]
        return bio_sequence

    @classmethod
    def tags2spans(cls, tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "I":
                if start is None:
                    start = i
            elif t == "E":
                if start is not None:
                    spans.append((start, i + 1))
                else:
                    # error case: expected B or I before
                    spans.append((i, i + 1))

                start = None
            elif t == "O":
                if start is not None:
                    # error case: expected E before
                    spans.append((start, i))
                start = None

        if start is not None:
            spans.append((start, len(tags)))
        return spans


class SafeIOE2Scheme(SafeDecodingScheme, IOE2Scheme):
    pass


class IOBESScheme(AbstractTaggingScheme):
    index2tag, tag2index = DataTools.get_mappings(["O", "B", "I", "E", "S"])

    # size = len(index2tag)

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        tag_sequence = ["O"] * length
        for s, e in spans:
            if e - s == 1:
                tag_sequence[s] = "S"
            elif e - s > 1:
                tag_sequence[s:e] = ["B"] + ["I"] * (e - s - 2) + ["E"]
            else:
                raise ValueError("Invalid span: ({}-{})".format(s, e))
        return tag_sequence

    @classmethod
    def tags2spans(cls, tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "B":
                if start is not None:
                    # error case: expected O or E or S before
                    spans.append((start, i))
                start = i
            elif t == "I":
                if start is None:
                    # error case: expected B before
                    start = i
            elif t == "E":
                if start is not None:
                    spans.append((start, i + 1))
                else:
                    # error case: expected S here
                    spans.append((i, i + 1))
                start = None
            elif t == "S":
                if start is not None:
                    # error case: expected E before
                    spans.append((start, i))
                spans.append((i, i + 1))
            elif t == "O":
                if start is not None:
                    # error case: expected E before
                    spans.append((start, i))
                start = None

        if start is not None:
            spans.append((start, len(tags)))
        return spans


class SafeIOBESScheme(SafeDecodingScheme, IOBESScheme):
    pass


class IOBExScheme(AbstractTaggingScheme):
    index2tag, tag2index = DataTools.get_mappings(["O", "I", "B", "E"])

    # size = len(index2tag)

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        tag_sequence = ["O"] * length
        for s, e in spans:
            if e - s == 1:
                tag_sequence[s] = "S"
            elif e - s > 1:
                tag_sequence[s:e] = ["B"] + ["I"] * (e - s - 2) + ["E"]
            else:
                raise ValueError("Invalid span: ({}-{})".format(s, e))
        return tag_sequence

    @classmethod
    def tags2spans(cls, tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "B":
                if start is not None:
                    # error case: expected O or E or S before
                    spans.append((start, i))
                start = i
            elif t == "I":
                if start is None:
                    # error case: expected B before
                    start = i
            elif t == "E":
                if start is not None:
                    spans.append((start, i + 1))
                else:
                    # error case: expected S here
                    spans.append((i, i + 1))
                start = None
            elif t == "S":
                if start is not None:
                    # error case: expected E before
                    spans.append((start, i))
                spans.append((i, i + 1))
            elif t == "O":
                if start is not None:
                    # error case: expected E before
                    spans.append((start, i))
                start = None

        if start is not None:
            spans.append((start, len(tags)))
        return spans

    @classmethod
    def tags2encoding(cls, tag_sequence):
        encoding = numpy.zeros((len(tag_sequence), cls.size()))
        for i, tag in enumerate(tag_sequence):
            if tag == "O":
                vec = [1, 0, 0, 0]
            elif tag == "I":
                vec = [0, 1, 0, 0]
            elif tag == "B":
                vec = [0, 1, 1, 0]
            elif tag == "E":
                vec = [0, 1, 0, 1]
            elif tag == "S":
                vec = [0, 1, 1, 1]
            else:
                raise ValueError("Unexpected tag: {}".format(tag))
            encoding[i, :] = vec
        return encoding

    @classmethod
    def encoding2tags(cls, encoding):
        encoding = numpy.array(encoding)
        tags = []
        for vec in encoding:
            outside = vec[cls.tag2index["O"]]
            inside = vec[cls.tag2index["I"]]
            begin = vec[cls.tag2index["B"]]
            end = vec[cls.tag2index["E"]]

            if outside > 0.5:
                tags.append("O")
            else:
                if inside > 0.5:
                    if begin > 0.5 and end > 0.5:
                        tags.append("S")
                    elif begin > 0.5 and end < 0.5:
                        tags.append("B")
                    elif end > 0.5 and begin < 0.5:
                        tags.append("E")
                    else:
                        tags.append("I")
                else:
                    raise ValueError("Unexpected encoding: {}".format(vec))

        return tags


class BinaryScheme(AbstractTaggingScheme):
    index2tag, tag2index = DataTools.get_mappings(["O", "I"])

    # size = len(index2tag)

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        tag_sequence = ["O"] * length
        for s, e in spans:
            tag_sequence[s:e] = ["I"] * (e - s)

        return tag_sequence

    @classmethod
    def tags2spans(cls, tags):
        spans = []
        start = None
        for i, t in enumerate(tags):
            if t == "I":
                if start is None:
                    start = i
            elif t == "O":
                if start is not None:
                    spans.append((start, i))
                start = None

        if start is not None:
            spans.append((start, len(tags)))
        return spans


class IOB2Scheme(AbstractTaggingScheme):
    index2tag, tag2index = DataTools.get_mappings(["O", "B", "I"])

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        bio_sequence = ["O"] * length
        for s, e in spans:
            bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
        return bio_sequence

    @classmethod
    def tags2spans(cls, tags):
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


class SafeIOB2Scheme(SafeDecodingScheme, IOB2Scheme):
    pass

    # @classmethod
    # def encoding2tags(cls, encoding):
    #     encoding = numpy.array(encoding)
    #
    #     inside_scores = encoding[:, cls.tag2index["B"]] + encoding[:, cls.tag2index["I"]]
    #
    #     outside_scores = encoding[:, cls.tag2index["O"]]
    #     binary_tags = list(inside_scores > outside_scores)
    #
    #     tags = ["I" if b else "O" for b in binary_tags]
    #     return tags


#################################
class IOBScheme(object):
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
    def safe_encoding2tag_sequence(encoding):
        encoding = numpy.array(encoding)
        inside_scores = encoding[:, IOBScheme.tag2index["B"]] + encoding[:, IOBScheme.tag2index["I"]]
        outside_scores = encoding[:, IOBScheme.tag2index["O"]]
        binary_tags = list(inside_scores > outside_scores)
        tags = ["I" if b else "O" for b in binary_tags]
        return tags

    @staticmethod
    def encoding2tags(encoding, safe=False):
        if safe:
            return IOBScheme.safe_encoding2tag_sequence(encoding)
        else:
            return encoding2tag_sequence(encoding, IOBScheme.index2tag)

    @staticmethod
    def encoding2spans(encoding, safe=False):
        return IOBScheme.tags2spans(IOBScheme.encoding2tags(encoding, safe=safe))

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
        colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
        return colored_text

    @staticmethod
    def visualize_tags(elements, tags, spacer=u""):
        cm = dict()
        cm["B"] = Fore.GREEN  # beginning
        cm["I"] = Fore.BLUE  # inside
        cm["O"] = ""  # outside
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda x: cm[x[1]] + x[0] + Fore.RESET, zip(elements, tags)))
        return colored_text

    @staticmethod
    def visualize_spans(elements, spans, spacer=u""):
        tags = IOBScheme.spans2tags(len(elements), spans)
        return IOBScheme.visualize_tags(elements, tags, spacer)

    def __str__(self):
        return IOBScheme.name

    def __repr__(self):
        return IOBScheme.name


class IOB2XScheme:
    index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
    size = len(index2tag)
    name = "IOB2X"

    @classmethod
    def spans2tags(cls, length, spans):
        sorted(spans, key=lambda s: s[0])
        bio_sequence = ["O"] * length
        for s, e in spans:
            bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
        return bio_sequence

    @classmethod
    def tags2spans(cls, tags):
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

    @classmethod
    def spans2encoding(cls, length, spans):
        return cls.tags2encoding(cls.spans2tags(length, spans))

    @classmethod
    def tags2encoding(cls, tag_sequence):
        n_tags = len(cls.tag2index)
        n_elements = len(tag_sequence)
        encoding = numpy.zeros((n_elements, n_tags))
        for i, t in enumerate(tag_sequence):
            if t == "B":
                encoding[i, :] = [1, 1, 0]
            elif t == "I":
                encoding[i, :] = [0, 1, 0]
            elif t == "O":
                encoding[i, :] = [0, 0, 1]

        return encoding

    @classmethod
    def encoding2tags(cls, encoding, safe=False):
        encoding = numpy.array(encoding)
        tags = []
        for vec in encoding:
            begin = vec[cls.tag2index["B"]]
            inside = vec[cls.tag2index["I"]]
            outside = vec[cls.tag2index["O"]]

            if safe:
                inside += begin

            if inside > outside:
                if begin >= 0.5:
                    tags.append("B")
                else:
                    tags.append("I")
            else:
                tags.append("O")

        return tags

    @classmethod
    def encoding2spans(cls, encoding, safe=False):
        return cls.tags2spans(cls.encoding2tags(encoding, safe=safe))

    @classmethod
    def visualize_encoding(cls, elements, bio, onehot=True, spacer=u""):
        b = Fore.GREEN  # beginning
        i = Fore.BLUE  # inside
        o = Fore.BLACK  # outside
        BI = [b, i, o]
        if onehot:
            bio_indices = numpy.argmax(bio, axis=1)
        else:
            bio_indices = bio
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
        return colored_text


# class IOB2Scheme:
#     index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
#     size = len(index2tag)
#     name = "IOB2"
#
#     @staticmethod
#     def spans2tags(length, spans):
#         sorted(spans, key=lambda s: s[0])
#         bio_sequence = ["O"] * length
#         for s, e in spans:
#             bio_sequence[s:e] = ["B"] + ["I"] * (e - s - 1)
#         return bio_sequence
#
#     @staticmethod
#     def tags2spans(tags):
#         spans = []
#         start = None
#         for i, t in enumerate(tags):
#             if t == "B":
#                 if start is not None:
#                     spans.append((start, i))
#                 start = i
#             elif t == "I":
#                 if start is None:
#                     start = i
#             elif t == "O":
#                 if start is not None:
#                     spans.append((start, i))
#                 start = None
#
#         if start:
#             spans.append((start, len(tags)))
#         return spans
#
#     @staticmethod
#     def spans2encoding(length, spans):
#         return IOB2Scheme.tags2encoding(IOB2Scheme.spans2tags(length, spans))
#
#     @staticmethod
#     def tags2encoding(tags):
#         return tag_sequence2encoding(tags, IOB2Scheme.tag2index)
#
#     @staticmethod
#     def safe_encoding2tag_sequence(encoding):
#         encoding = numpy.array(encoding)
#         inside_scores = encoding[:, IOB2Scheme.tag2index["B"]] + encoding[:, IOB2Scheme.tag2index["I"]]
#         outside_scores = encoding[:, IOB2Scheme.tag2index["O"]]
#         binary_tags = list(inside_scores > outside_scores)
#         tags = ["I" if b else "O" for b in binary_tags]
#         return tags
#
#     @staticmethod
#     def encoding2tags(encoding, safe=False):
#         if safe:
#             return IOB2Scheme.safe_encoding2tag_sequence(encoding)
#         else:
#             return encoding2tag_sequence(encoding, IOB2Scheme.index2tag)
#
#     @staticmethod
#     def encoding2spans(encoding, safe=False):
#         return IOB2Scheme.tags2spans(IOB2Scheme.encoding2tags(encoding, safe=safe))
#
#     @staticmethod
#     def visualize_encoding(elements, bio, onehot=True, spacer=u""):
#         b = Fore.GREEN  # beginning
#         i = Fore.BLUE  # inside
#         o = Fore.BLACK  # outside
#         BI = [b, i, o]
#         if onehot:
#             bio_indices = numpy.argmax(bio, axis=1)
#         else:
#             bio_indices = bio
#         # outside should not be colored at all
#         colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
#         return colored_text
#
#     def __str__(self):
#         return IOB2Scheme.name
#
#     def __repr__(self):
#         return IOB2Scheme.name


class CountScheme:
    index2tag, tag2index = DataTools.get_mappings(["B", "I", "O"])
    size = len(index2tag)
    name = "count"

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
        colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
        return colored_text

    @staticmethod
    def visualize_tags(elements, tags, spacer=u""):
        cm = dict()
        cm["B"] = Fore.GREEN  # beginning
        cm["I"] = Fore.BLUE  # inside
        cm["O"] = ""  # outside
        # outside should not be colored at all
        colored_text = spacer.join(map(lambda x: cm[x[1]] + x[0] + Fore.RESET, zip(elements, tags)))
        return colored_text

    @staticmethod
    def visualize_spans(elements, spans, spacer=u""):
        tags = IOBScheme.spans2tags(len(elements), spans)
        return IOBScheme.visualize_tags(elements, tags, spacer)

    def __str__(self):
        return IOBScheme.name

    def __repr__(self):
        return IOBScheme.name


#####################################

def get_tagging_scheme(name):
    tagging_scheme = getattr(sys.modules[__name__], name)

    if tagging_scheme:
        return tagging_scheme
    else:
        lname = name.lower()
        if name == IOBScheme.__name__ or lname == "iob" or lname == "bio":
            return IOBScheme
        elif name == IOB2Scheme.__name__ or lname == "iob2" or lname == "bio2":
            return IOB2Scheme
        elif name == BinaryScheme.__name__ or lname == "binary":
            return BinaryScheme
        elif name == CountScheme.__name__ or lname == "count":
            return CountScheme


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
    colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
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
    colored_text = spacer.join(map(lambda x: BI[x[1]] + x[0] + Fore.RESET, zip(elements, bio_indices)))
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
    return list(map(lambda a: (a.start, a.end), annotations))


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
            matched_spans.append(functools.reduce(lambda n, t: (min(t[0], s[0]), max(t[1], s[1])), matching_tokens, s))
    return matched_spans


def is_touching(prev_span, span):
    return prev_span[1] == span[0]
