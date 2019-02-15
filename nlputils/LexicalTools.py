import re
from operator import itemgetter
from typing import Sequence, TypeVar, Callable, Tuple, List, Dict

import nltk
import numpy
from nltk import FreqDist
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

from nlputils import DataTools

__author__ = 'sjebbara'
# Visualize tokenization pattern for sentence:
#   Test sentence, test Test, test-designed :-P.  testTest test.Test test.test: 1-23 ('test')

tokenization_patterns = dict()
# Test  sentence  ,  test  Test  ,  test  -  designed  :  -  P  .  testTest  test  .  Test  test  .  test  :  1  -  23  (  '  test  '  )
tokenization_patterns["standard"] = re.compile("\w+|[^\w\s]", re.UNICODE)
# Test     sentence  ,     test     Test  ,     test  -  designed     :  -  P  .      testTest     test  .  Test     test  .  test  :     1  -  23     (  '  test  '  )
tokenization_patterns["include_all"] = re.compile("\w+|\s+|[^\w\s]", re.UNICODE)
# Test  sentence  ,  test  Test  ,  test-designed  :-  P  .  testTest  test  .  Test  test  .  test  :  1-23  ('  test  ')
tokenization_patterns["inner_symbol"] = re.compile("\d+[.]\d+|\w+[-']\w+|\w+|[^\w\s]+", re.UNICODE)
# Test  sentence  ,  test  Test  ,  test-designed  :-  P  .  testTest  test  .  Test  test  .  test  :  1-23  ('  test  ')
tokenization_patterns["advanced"] = re.compile("\d+[./]\d+|\w+[-']\w+|\w+|[^\w\s]+", re.UNICODE)
# Test  sentence  ,  test  Test  ,  test-designed  :  -  P  .  testTest  test  .  Test  test  .  test  :  1-23  (  '  test  '  )
tokenization_patterns["advanced2"] = re.compile("\d+[./]\d+|\w+[-']\w+|\w+|[^\w\s]", re.UNICODE)
tokenization_patterns["advanced2-no-punc"] = re.compile("\d+[./]\d+|\w+[-']\w+|\w+|[^\w\s]", re.UNICODE)
tokenization_patterns["advanced3"] = re.compile("\d+[./]\d+|\w+[-']\w+|\w+", re.UNICODE)

# stanford_sentence_splitter = nltk.tokenize.stanford.StanfordTokenizer(
#     os.path.expanduser("~/programs/stanford-corenlp-full-2015-12-09/stanford-corenlp-3.6.0.jar"))
# sentence_splitter = nltk.tokenize.PunktSentenceTokenizer()
sentence_boundary_regex = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(\s+|\b)(?!\d)")

POS_TAG_SET = (
    "<PAD>", "<UNK>", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNS", "NNP", "NNPS",
    "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "WDT", "WP", "WP$", "WRB", "#", "$", "''", "``", ",", ".", ":")

pos_vocabulary = DataTools.Vocabulary()
pos_vocabulary.init_from_word_list(POS_TAG_SET)
pos_vocabulary.set_padding(0)
pos_vocabulary.set_unknown(1)

CHUNK_TAG_SET = (
    "B-PRT", "I-PRT", "B-LST", "I-LST", "B-ADVP", "I-ADVP", "B-VP", "I-VP", "B-CONJP", "I-CONJP", "B-NP", "I-NP",
    "B-ADJP", "I-ADJP", "B-PP", "I-PP", "B-INTJ", "I-INTJ", "B-SBAR", "I-SBAR", "B-UCP", "I-UCP", "O")

STOPWORDS = set(stopwords.words("english"))

NEGATION_WORDS = {'barely', 'hardly', 'non', 'nowhere', 'nor', 'no', 'cannot', 'not', 'neither', 'scarcely', "n't", 'never', 'none', 'nobody',
                  'nothing'}


def get_pos_tag_vocabulary():
    pos_vocabulary = DataTools.Vocabulary()
    pos_vocabulary.init_from_word_list(POS_TAG_SET)
    pos_vocabulary.set_padding(0)
    pos_vocabulary.set_unknown(1)
    return pos_vocabulary


def get_chunk_tag_vocabulary():
    chunk_vocabulary = DataTools.Vocabulary()
    chunk_vocabulary.init_from_word_list(CHUNK_TAG_SET)
    chunk_vocabulary.add_padding("<pad>", 0)
    return chunk_vocabulary


whitespace_reduce_regex = re.compile(r"\s+")
whitespace_regex = re.compile(r"\s")

simple_url_regex = re.compile("(https?|ftp):\/\/[^\s/$.?#].[^\s]*")

DEFAULT_CHAR_SET = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.;:!?-_\"\'\\/()[]{}<>=+*$€£¥%&§@#")


def get_char_vocabulary():
    char_vocabulary = DataTools.Vocabulary()
    char_vocabulary.init_from_word_list(DEFAULT_CHAR_SET)
    char_vocabulary.add_padding("<0>", 0)
    char_vocabulary.add_unknown("<?>", 1)
    return char_vocabulary


def lower(x):
    return x.lower()


def clean_whitespaces(x, reduce_whitespaces=False):
    if reduce_whitespaces:
        return re.sub(whitespace_reduce_regex, " ", x)
    else:
        return re.sub(whitespace_regex, " ", x)


def tokenize(line, tokenization_style="standard", stopwords=None):
    s, e, tokens = tokenization(line, tokenization_style=tokenization_style, stopwords=stopwords)
    return tokens


def tokenization(text, tokenization_style="standard", stopwords=None):
    starts = []
    ends = []
    tokens = []
    if type(tokenization_style) == str:
        if tokenization_style in tokenization_patterns:
            tokenization_pattern = tokenization_patterns[tokenization_style]
        else:
            tokenization_pattern = re.compile(tokenization_style, re.UNICODE)
    else:
        tokenization_pattern = tokenization_style
    for m in tokenization_pattern.finditer(text):
        start = m.start()
        end = m.end()
        token = m.group()
        if stopwords is None or token not in stopwords:
            starts.append(start)
            ends.append(end)
            tokens.append(token)

    return starts, ends, tokens


def get_tokens(text, spans):
    tokens = [text[start:end] for start, end in spans]
    return tokens


def token_features(tokens, spans):
    import AnnotationTools
    features = []
    for i, (token, span) in enumerate(zip(tokens, spans)):
        if i > 0:
            prev_span = spans[i - 1]
            separated_to_prev_token = AnnotationTools.is_touching(prev_span, span)
        else:
            separated_to_prev_token = True

        init_cap = starts_with_capital_character(token)
        all_cap = capital_characters_only(token)
        contains_cap = contains_capital_character(token)

        all_number = number_characters_only(token)
        contains_number = contains_number_character(token)

        f = numpy.array([init_cap, all_cap, contains_cap, all_number, contains_number, separated_to_prev_token])
        features.append(f)
    return features


def starts_with_capital_character(token):
    return token[0].isupper()


def capital_characters_only(token):
    return token.isupper()


def contains_capital_character(token):
    return any([c.isupper() for c in token])


def number_characters_only(token):
    return token.isdigit()


def contains_number_character(token):
    return any([c.isdigit() for c in token])


is_punctuation_regex = re.compile(r"\W+")


def is_punctuation(token):
    return is_punctuation_regex.match(token) is not None


def _get_pos_tagger():
    return nltk.tag.StanfordPOSTagger(
        "/home/sjebbara/programs/stanford-postagger-2017-06-09/models/english-left3words-distsim.tagger",
        path_to_jar="/home/sjebbara/programs/stanford-postagger-2017-06-09/stanford-postagger.jar")


def get_pos_tags(sentences, as_index=True):
    tagger = _get_pos_tagger()
    tagged_sentences = tagger.tag_sents(sentences)

    if as_index:
        return [[pos_vocabulary.get_index(p) for w, p in s] for s in tagged_sentences]
    return [[p for w, p in s] for s in tagged_sentences]


def split_sentences(text):
    # sentences = sentence_splitter.tokenize(text)
    starts = [0]
    ends = []
    sentences = []

    for m in sentence_boundary_regex.finditer(text):
        boundary_start = m.start()
        boundary_end = m.end()
        ends.append(boundary_start)
        starts.append(boundary_end)

    ends.append(len(text))
    sentences = [text[s:e] for s, e in zip(starts, ends)]
    return sentences, starts, ends


def filter_stopwords(stopwords, starts, ends, tokens):
    tokens_filtered = []
    starts_filtered = []
    ends_filtered = []

    shift = 0
    for t, s, e in zip(tokens, starts, ends):
        if t in stopwords:
            # shift += 1
            pass
        else:
            tokens_filtered.append(t)
            starts_filtered.append(s - shift)
            ends_filtered.append(e - shift)
    return starts_filtered, ends_filtered, tokens_filtered


def pad_sequence(sequence: List, n, pad_start=None, pad_end=None):
    start_padding = [pad_start] * (n - 1) * (pad_start is not None)
    end_padding = [pad_end] * (n - 1) * (pad_end is not None)
    sequence = start_padding + sequence + end_padding
    return sequence


def get_n_grams(tokens: List, n, pad_start=None, pad_end=None):
    tokens = pad_sequence(tokens, n, pad_start, pad_end)
    return zip(*[tokens[i:] for i in range(n)])


def get_n_grams_no_pad(tokens: List, n):
    return zip(*[tokens[i:] for i in range(n)])


def is_negation(token):
    if token in NEGATION_WORDS:
        return True
    elif token.endswith("'nt"):
        return True
    else:
        return False


def get_character_word_ngrams(tokens, n, pad_start=None, pad_end=None):
    char_ngrams = [get_n_grams(list(t), n, pad_start, pad_end) for t in tokens]
    return char_ngrams


T = TypeVar("T")


# def levenshtein_distance(sequence1: Sequence[T], sequence2: Sequence[T],
#                          insertion_cost: Callable[[T], float] = None,
#                          deletion_cost: Callable[[T], float] = None,
#                          substitution_cost: Callable[[T, T], float] = None):
#     insertion_cost = insertion_cost or (lambda x: 1.)
#     deletion_cost = deletion_cost or (lambda x: 1.)
#     substitution_cost = substitution_cost or (lambda x, y: float(x != y))
#
#     if sequence1 == sequence2:
#         return 0
#     if len(sequence1) < len(sequence2):
#         sequence1, sequence2 = sequence2, sequence1
#     if not sequence1:
#         return len(sequence2)
#
#     previous_row = range(len(sequence2) + 1)
#     for i, element1 in enumerate(sequence1):
#         current_row = [i + 1]
#         for j, element2 in enumerate(sequence1):
#             insertions = previous_row[j + 1] + insertion_cost(element2)
#             deletions = current_row[j] + deletion_cost(element2)
#             substitutions = previous_row[j] + substitution_cost(element1, element2)
#             current_row.append(min(insertions, deletions, substitutions))
#         previous_row = current_row
#     return previous_row[-1]


def levenshtein_distance(sequence1: Sequence[T], sequence2: Sequence[T],
                         insertion_cost: Callable[[T], float] = None,
                         deletion_cost: Callable[[T], float] = None,
                         substitution_cost: Callable[[T, T], float] = None,
                         transposition_cost: Callable[[T, T], float] = None,
                         damerau: bool = False):
    insertion_cost = insertion_cost or (lambda x: 1.)
    deletion_cost = deletion_cost or (lambda x: 1.)
    substitution_cost = substitution_cost or (lambda x, y: float(x != y))
    transposition_cost = transposition_cost or (lambda x, y: float(x != y))

    d = {}
    lenstr1 = len(sequence1)
    lenstr2 = len(sequence2)
    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        s_i = sequence1[i]
        for j in range(lenstr2):
            s_j = sequence2[j]
            d[(i, j)] = min(
                d[(i - 1, j)] + deletion_cost(s_i),  # deletion
                d[(i, j - 1)] + insertion_cost(s_j),  # insertion
                d[(i - 1, j - 1)] + substitution_cost(s_i, s_j),  # substitution
            )
            if damerau and i and j and s_i == sequence2[j - 1] and sequence1[i - 1] == s_j:
                d[(i, j)] = min(d[(i, j)],
                                d[i - 2, j - 2] + transposition_cost(s_i, s_j)
                                )  # transposition

    return d[lenstr1 - 1, lenstr2 - 1]


class EmbeddingDistance:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._distances = dict()

    def __call__(self, word1, word2):
        if (word1, word2) in self._distances:
            return self._distances[(word1, word2)]
        else:
            d = self._distance(word1, word2)
            self._distances[(word1, word2)] = d
            return d

    def _distance(self, word1, word2):
        v1 = self.embeddings.get_vector(word1)
        v2 = self.embeddings.get_vector(word2)
        distance = cosine(v1, v2)
        return distance


class EmbedditDistance:
    def __init__(self, embeddings):
        self.embedding_distance = EmbeddingDistance(embeddings)

    def __call__(self, tokens1: Sequence[str], tokens2: Sequence[str]):
        return levenshtein_distance(tokens1, tokens2,
                                    insertion_cost=self.insertion_cost,
                                    deletion_cost=self.deletion_cost,
                                    substitution_cost=self.substitution_cost,
                                    transposition_cost=self.transposition_cost,
                                    damerau=True)

    def insertion_cost(self, token: str) -> float:
        if is_negation(token):
            return 1.5
        elif token in STOPWORDS:
            return 0.5
        else:
            return 1

    def deletion_cost(self, token: str) -> float:
        return self.insertion_cost(token)

    def substitution_cost(self, token1: str, token2: str) -> float:
        if (token1 in NEGATION_WORDS) ^ (token2 in NEGATION_WORDS):
            return self.embedding_distance(token1, token2) + 0.5
        else:
            return self.embedding_distance(token1, token2)

    def transposition_cost(self, token1: str, token2: str):
        return self.substitution_cost(token1, token2) + 0.2


NgramT = Tuple[str, ...]
SegmentationT = Tuple[float, List[NgramT]]


class NGramSegmenter:
    def __init__(self, ngram_sizes: Sequence[int], pad_start: str = None, pad_end: str = None):
        self.ngram_sizes = ngram_sizes
        self.pad_start = pad_start
        self.pad_end = pad_end
        self.ngram_fd = None  # type: FreqDist

    def fit(self, tokenized_sequences: Sequence[List[str]], labels=None):
        self.ngram_fd = FreqDist()

        for tokens in tokenized_sequences:
            tokens = [self.pad_start] + tokens + [self.pad_end]
            for n in self.ngram_sizes:
                token_ngrams = get_n_grams_no_pad(tokens, n)
                self.ngram_fd.update(token_ngrams)

    def transform(self, tokenized_sequences: Sequence[Sequence[str]], delimiter: str = None):
        segmented_sequences = []
        for tokens in tokenized_sequences:
            segmentation = self.segment_sequence(tokens, delimiter=delimiter)
            segmented_sequences.append(segmentation)
        return segmented_sequences

    def _ngram_score(self, ngram: Tuple[str, ...]):
        score = self.ngram_fd.freq(ngram)
        score = numpy.power(score, 1 / len(ngram))

        return score

    def segment_sequence(self, tokens: Sequence[str], delimiter: str = None) -> SegmentationT:
        tokens = list(tokens)
        tokens = [self.pad_start] + tokens + [self.pad_end]
        score, segmentation = self._segment_sequence(tuple(tokens), {})
        if delimiter is not None:
            segmentation = [delimiter.join(ngram) for ngram in segmentation]

        return score, segmentation

    def _segment_sequence(self, tokens: Tuple[str, ...], memory: Dict[Tuple[str], SegmentationT]) -> SegmentationT:
        if tokens in memory:
            return memory[tokens]
        elif len(tokens) == 0:
            return 0, []
        else:
            segmentations = []
            for n in self.ngram_sizes:
                ngram = tuple(tokens[:n])
                ngram_score = self._ngram_score(ngram)
                sub_score, sub_segmentation = self._segment_sequence(tokens[n:], memory)

                score = ngram_score + sub_score
                segmentation = [ngram] + sub_segmentation
                segmentations.append((score, segmentation))

                if len(tokens) <= n:
                    break

            segmentations = sorted(segmentations, key=itemgetter(0), reverse=True)
            best_segmentation = segmentations[0]
            memory[tokens] = best_segmentation
            return best_segmentation
