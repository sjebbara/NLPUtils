
import re
import os
import numpy
import nltk
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


def get_n_grams(tokens, n, pad_start=None, pad_end=None):
    start_padding = [pad_start] * (n - 1) * (pad_start is not None)
    end_padding = [pad_end] * (n - 1) * (pad_end is not None)
    tokens = start_padding + tokens + end_padding
    return zip(*[tokens[i:] for i in range(n)])


negation_words = set("no cannot not none nothing nowhere neither nor never nobody hardly scarcely barely".split())


def is_negation(token):
    if token in negation_words:
        return True
    elif token.endswith("'nt"):
        return True
    else:
        return False


def get_character_word_ngrams(tokens, n, pad_start=None, pad_end=None):
    char_ngrams = [get_n_grams(list(t), n, pad_start, pad_end) for t in tokens]
    return char_ngrams
