import io
import json
import os
import numpy
import LearningTools
from sklearn.base import BaseEstimator, TransformerMixin
from xml.sax.saxutils import unescape
from collections import defaultdict, Counter

__author__ = 'sjebbara'

html_unescape_table = {"&quot;": '"', "&apos;": "'", "&#34;": '"'}

polarity2scalar_map = dict()
polarity2scalar_map["positive"] = 1.
polarity2scalar_map["conflict"] = 0.
polarity2scalar_map["negative"] = -1.


def polarity2scalar(polarity):
    return polarity2scalar_map[polarity]


def scalar2polarity(scalar):
    threshold = 0.5
    if -1 <= scalar <= -threshold:
        return "negative"
    elif -threshold < scalar <= threshold:
        return "conflict"
    elif threshold < scalar <= 1:
        return "positive"


polarities5 = "negative weak-negative neutral weak-positive positive".split()
polarity2scalar5_map = dict(zip(polarities5, numpy.linspace(-1, 1, len(polarities5))))


def polarity2scalar5(polarity):
    return polarity2scalar5_map[polarity]


def scalar2polarity5(scalar):
    for p, v in zip(polarities5, numpy.linspace(-1, 1, len(polarities5) + 1)[1:]):
        if scalar <= v:
            return p


class AbstractText(object):
    def __init__(self):
        self.text_id = None
        self.text = None

        self.sentences = None
        self.tokens = None
        self.starts = None
        self.ends = None

        self.token_pos_tags = []

    def copy(self):
        return None

    def init_from_document(self, d):
        self.text_id = d.text_id
        self.text = d.text

        self.tokens = d.tokens
        self.starts = d.starts
        self.ends = d.ends

        self.sentences = []
        if d.sentences is not None:
            for s in d.sentences:
                new_s = s.copy()
                self.sentences.append(new_s)

    def tokenize(self, tokenization_style="advanced", stopwords=None):
        from nlputils import LexicalTools
        self.starts, self.ends, self.tokens = LexicalTools.tokenization(self.text,
                                                                        tokenization_style=tokenization_style)

        if stopwords:
            self.starts, self.ends, self.tokens = LexicalTools.filter_stopwords(stopwords, self.starts, self.ends,
                                                                                self.tokens)
        if self.sentences is not None:
            for s in self.sentences:
                s.tokenize(tokenization_style, stopwords)

    def split_into_sentences(self):
        from nlputils import LexicalTools
        sentence_documents = []
        sentences, starts, ends = LexicalTools.split_sentences(self.text)

        for i, (s, start, end) in enumerate(zip(sentences, starts, ends)):
            d_sentence = self.copy()
            d_sentence.text_id = self.text_id + "#" + str(i)
            d_sentence.text = s
            sentence_documents.append(d_sentence)
        self.sentences = sentence_documents
        return sentence_documents


class DataSample:
    def __init__(self):
        pass


class Vocabulary:
    def __init__(self):
        self.vocab = set()
        self.counts = Counter()
        self.index2word = []
        self.word2index = dict()

    def set_padding(self, index):
        self.padding_index = index
        self.padding_word = self.get_word(index)

    def add_padding(self, word, index):
        self.add(word, index)
        self.set_padding(index)

    def set_unknown(self, index):
        self.unknown_index = index
        self.unknown_word = self.get_word(index)

    def add_unknown(self, word, index):
        self.add(word, index)
        self.set_unknown(index)

    def init_from_vocab(self, vocab):
        self.vocab = set(vocab)

        tmp_vocab = list(self.vocab)
        self.index2word, self.word2index = get_mappings(tmp_vocab)
        self.counts = Counter(self.vocab)

    def init_from_mapping(self, index2word, word2index):
        self.index2word = index2word
        self.word2index = word2index
        self.vocab = set(self.word2index.keys())
        self.counts = Counter(self.vocab)

    def init_from_word_list(self, vocab_list, counts=None):
        self.index2word, self.word2index = get_mappings(vocab_list)
        self.vocab = set(self.word2index.keys())
        if counts:
            self.counts = counts
        else:
            self.counts = Counter(self.vocab)

    def init_from_counts(self, counts):
        self.counts = counts

        self.index2word, self.word2index = get_mappings(self.counts.keys())
        self.vocab = set(self.word2index.keys())

    def init_from_gensim(self, w2v):
        self.index2word, self.word2index = get_mappings(w2v.index2word)
        self.vocab = set(self.word2index.keys())
        self.counts = Counter(dict([(w, v.count) for w, v in w2v.vocab.iteritems()]))

    def __len__(self):
        return len(self.vocab)

    def __contains__(self, word):
        return word in self.word2index

    def __str__(self):
        s = u"#Vocab: %d" % (len(self.vocab))
        if hasattr(self, "padding_word") and self.padding_word:
            s += u"\n  padding: %d = '%s'" % (
                self.word2index[self.padding_word], self.index2word[self.word2index[self.padding_word]])
        if hasattr(self, "unknown_word") and self.unknown_word:
            s += u"\n  unknown: %d = '%s'\n" % (
                self.word2index[self.unknown_word], self.index2word[self.word2index[self.unknown_word]])

        if len(self.index2word) <= 10:
            s += u"[{}]".format(", ".join([u"'{}'".format(w) for w in self.index2word]))
        else:
            s += u"[{}, ... , {}]".format(", ".join([u"'{}'".format(w) for w in self.index2word[:5]]),
                                          ", ".join([u"'{}'".format(w) for w in self.index2word[-5:]]))

        return s.encode("utf-8")

    def most_common(self, top_k=None):
        return [w for w, c in self.counts.most_common(top_k)]

    def words(self):
        return self.vocab

    def get_index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            if hasattr(self, "unknown_index") and self.unknown_index is not None:
                return self.unknown_index
            else:
                raise ValueError("Unknown word received and no value specified for that.")

    def get_indices(self, words, drop_unknown=False):
        if drop_unknown or not hasattr(self, "unknown_index"):
            return values2indices(words, self.word2index)
        else:
            return values2indices(words, self.word2index, default=self.unknown_index)

    def indices(self):
        return range(len(self.index2word))

    def get_word(self, index):
        return self.index2word[index]

    def get_words(self, indices, drop_unknown=False):
        if drop_unknown or not hasattr(self, "unknown_word"):
            return indices2values(indices, self.index2word)
        else:
            return indices2values(indices, self.index2word, default=self.unknown_word)

    def to_one_hot(self, word):
        v = numpy.zeros(len(self.index2word))
        v[self.get_index(word)] = 1
        return v

    def to_k_hot(self, words):
        v = numpy.zeros(len(self.index2word))
        for i, word in enumerate(words):
            v[self.get_index(word)] = 1
        return v

    def to_bow(self, words):
        v = numpy.zeros(len(self.index2word))
        for i, word in enumerate(words):
            v[self.get_index(word)] += 1
        return v

    def to_bbow(self, words):
        v = self.to_bow(words)
        v = (v > 0).astype(int)
        return v

    def to_one_hot_sequence(self, words):
        v = numpy.zeros((len(words), len(self.index2word)))
        for i, word in enumerate(words):
            v[i, self.get_index(word)] = 1
        return v

    def from_one_hot(self, one_hot):
        i = numpy.argmax(one_hot)
        word = self.get_word(i)
        return word

    def from_k_hot(self, k_hot, threshold=0.5):
        words = set(self.get_words(i for i, x in enumerate(k_hot) if x > threshold))
        return words

    def from_bow(self, bow, threshold=0.5):
        words = []
        for i, v in enumerate(bow):
            k = int(v)
            r = v - k
            if r > threshold:
                k += 1
            words += [self.get_word(i)] * k
        return words

    def from_bbow(self, bow, threshold=0.5):
        words = set(self.get_words(i for i, x in enumerate(bow) if x > threshold))
        return words

    def save(self, filepath):
        word_count_list = [(w, self.counts[w]) for w in self.index2word]

        def to_str(x):
            w, c = x
            return w + u" " + str(c)

        LearningTools.write_iterable(word_count_list, filepath, to_str)

    def load(self, filepath):
        def to_object(line):
            parts = line.split(" ")
            word = parts[0]
            count = int(parts[1])
            return word, count

        word_count_list = LearningTools.load_as_list(filepath, to_object=to_object)
        self.index2word = [w for w, c in word_count_list]
        # todo does this work?
        self.counts = Counter(dict(word_count_list))

        self.index2word, self.word2index = get_mappings(self.index2word)
        self.vocab = set(self.index2word)

    def add(self, word, index=None):
        if word in self.word2index:
            print("already there", word, self.word2index[word])
            return self.word2index[word]
        else:
            if index is None:
                return self.append(word)
            else:
                new_word_order = self.index2word[:index] + [word] + self.index2word[index:]
                self.counts[word] = 1
                self.init_from_word_list(new_word_order, self.counts)

                if hasattr(self, "unknown_index") and self.unknown_index is not None and index <= self.unknown_index:
                    self.unknown_index += 1
                    print("Moved {} to {}".format(self.unknown_word, self.unknown_index))
                if hasattr(self, "padding_index") and self.padding_index is not None and index <= self.padding_index:
                    self.padding_index += 1
                    print("Moved {} to {}".format(self.padding_word, self.padding_index))
                return index

    def append(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            self.vocab.add(word)
            index = len(self.index2word)
            self.index2word.append(word)
            self.word2index[word] = index
            self.counts[word] = 1
            return index


class Embedding:
    def __init__(self):
        self.vocabulary = None
        self.W = None

    def __len__(self):
        return len(self.vocabulary)

    def __contains__(self, word):
        return word in self.vocabulary

    def init(self, vocabulary, W, padding=None):
        assert isinstance(vocabulary, Vocabulary)
        if padding:
            padding_vector = numpy.zeros((1, W.shape[1]))
            self.W = numpy.concatenate((padding_vector, W), axis=0)
            self.vocabulary = vocabulary
            vocabulary.init_from_word_list([padding] + vocabulary.index2word)
        else:
            self.vocabulary = vocabulary
            self.W = W

    def init_from_gensim(self, w2v):
        self.vocabulary = Vocabulary()
        self.vocabulary.init_from_gensim(w2v)
        self.W = w2v.syn0

    def trim_embeddings(self, vocab_trim=None, top_k=None):
        counts = Counter()

        if vocab_trim:
            vocab_trim = set(vocab_trim)
            for w in vocab_trim:
                counts[w] = self.vocabulary.counts[w]
        else:
            vocab_trim = set()

        if top_k:
            counts.update(dict(self.vocabulary.counts.most_common(top_k)))
            vocab_trim.update(set([w for w, c in counts.iteritems()]))

        indices = self.vocabulary.get_indices(vocab_trim)
        self.W = self.W[indices]
        self.vocabulary.init_from_word_list(vocab_trim, counts)

    def get_vector(self, word):
        return self.W[self.vocabulary.get_index(word), :]

    def get_vectors(self, words, drop_unknown=False):
        indices = self.vocabulary.get_indices(words, drop_unknown)
        if len(indices) > 0:
            return self.W[indices, :]
        else:
            return numpy.zeros((0, self.W.shape[1]))

    def save(self, dirpath, embedding_name):
        embedding_filepath = os.path.join(dirpath, embedding_name + "_W.npy")
        vocab_filepath = os.path.join(dirpath, embedding_name + "_vocab.txt")
        numpy.save(embedding_filepath, self.W)
        self.vocabulary.save(vocab_filepath)

    def load(self, embedding_filepath, vocab_filepath):
        self.W = numpy.load(embedding_filepath)
        self.vocabulary = Vocabulary()
        self.vocabulary.load(vocab_filepath)

    def load_plain_text_file(self, filepath, top_k=None):
        words = []
        vectors = []
        with io.open(filepath) as f:
            for line in f:
                if len(words) >= top_k:
                    break
                parts = line.split(" ")
                words.append(parts[0])
                vectors.append(map(float, parts[1:]))

        vectors = numpy.array(vectors)
        vocabulary = Vocabulary()
        vocabulary.init_from_word_list(words)
        self.init(vocabulary, vectors)

    def add(self, word, index=None, vector=None, vector_init=None):
        if vector is None:
            if vector_init == "zeros":
                vector = numpy.zeros(self.W.shape[1:])
            elif vector_init == "mean":
                vector = numpy.mean(self.W, axis=0)
            elif vector_init == "uniform":
                m = numpy.mean(self.W, axis=0)
                s = numpy.std(self.W, axis=0)
                vector = numpy.random.rand(self.W.shape[1]) * s + m / 2
            elif vector_init == "normal":
                m = numpy.mean(self.W, axis=0)
                s = numpy.std(self.W, axis=0)
                vector = numpy.random.randn(self.W.shape[1]) * s + m

        vector = numpy.expand_dims(vector, axis=0)
        if index is None:
            self.vocabulary.add(word)
            self.W = numpy.append(self.W, vector, axis=0)
        else:
            self.vocabulary.add(word, index)
            self.W = numpy.concatenate((self.W[:index], vector, self.W[index:]), axis=0)

    def normalize(self):
        self.W = self.W / numpy.expand_dims(numpy.linalg.norm(self.W, axis=1), axis=1)


class WordIndexTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word_vocabulary=None):
        self.word_vocabulary = word_vocabulary

    def transform(self, tokenized_documents, drop_unknown=False, **transform_params):
        I = []
        max_len = 0
        for tokens in tokenized_documents:
            indices = self.word_vocabulary.get_indices(tokens, drop_unknown=drop_unknown)
            max_len = max(max_len, len(tokens))
            I.append(indices)

        I = [[self.word_vocabulary.padding_index] * (max_len - len(indices)) + list(indices) for indices in I]
        I = numpy.array(I)
        return I  # {"token_input": I}

    def fit(self, tokenized_documents, y=None, **transform_params):
        self.word_vocabulary = Vocabulary()
        vocab_counts = Counter()
        for tokens in tokenized_documents:
            vocab_counts.update(tokens)

        self.word_vocabulary.init_from_counts(vocab_counts)
        self.word_vocabulary.add_padding("<pad>", 0)
        self.word_vocabulary.add_unknown("<unk>", 1)
        return self


class LambdaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, function):
        self.function = function

    def transform(self, documents, **transform_params):
        return map(self.function, documents)

    def fit(self, X, y=None, **transform_params):
        return self


def html_unescape(text):
    return unescape(text, html_unescape_table)


def insert_and_get_index(index2word, word2index, word):
    if word not in word2index:
        index = len(word2index)
        index2word[index] = word
        word2index[word] = index
    else:
        index = word2index[word]
    return index


def train_test_split_indices(N, fraction, seed=None):
    if seed is not None:
        numpy.random.seed(seed)
    P = numpy.random.permutation(range(N))
    S = numpy.ceil(N * fraction)
    S = S.astype(int)
    I_train = P[:S]
    I_test = P[S:]
    return I_train, I_test


def cross_validation_split_indices(N, folds, seed=None):
    if seed is not None:
        numpy.random.seed(seed)
    P = numpy.random.permutation(range(N))

    S = numpy.array(numpy.linspace(0, N, folds + 1), dtype="int")
    I = list(map(lambda x: P[x[0]:x[1]], zip(S[:-1], S[1:])))
    return I


def cross_validation_split(documents, folds, seed=None):
    I = cross_validation_split_indices(len(documents), folds, seed=seed)
    cv_data = []
    for k_test in range(len(I)):
        test_documents = subset(documents, I[k_test])
        train_documents = []
        for k_train in range(len(I)):
            if k_test != k_train:
                train_documents += subset(documents, I[k_train])
        cv_data.append((train_documents, test_documents))
    return cv_data


def custom_split(documents, fraction, seed=None):
    I_train, I_test = train_test_split_indices(len(documents), fraction, seed=seed)
    train_documents = subset(documents, I_train)
    test_documents = subset(documents, I_test)
    return (train_documents, test_documents)


def one_of_k(I, K):
    X = numpy.zeros((I.shape[0], I.shape[1], K))
    for m in range(I.shape[0]):
        for n in range(I.shape[1]):
            X[m, n, I[m, n]] = 1
    return X


def subset(X, I):
    return [X[i] for i in I]


def values2indices(values, value2index, default=None):
    if default is None:
        vec = numpy.array([value2index[v] for v in values if v in value2index])
    else:
        vec = numpy.array([value2index[v] if v in value2index else default for v in values])

    # vec = numpy.expand_dims(vec, axis=0)
    return vec


def indices2values(indices, index2value, default=None):
    if default is None:
        val = [index2value[i] for i in indices if i < len(index2value)]
    else:
        val = [index2value[i] if i < len(index2value) else default for i in indices]

    return val


def get_mappings(vocab):
    index2value = list(vocab)
    value2index = dict((v, i) for i, v in enumerate(index2value))
    return index2value, value2index


def trim_embeddings(vocab, W, token2index):
    indices = values2indices(vocab, token2index)
    W_trim = W[indices]
    index2token_trim, token2index_trim = get_mappings(vocab)
    return W_trim, index2token_trim, token2index_trim


def add_padding_front(I):
    P = numpy.zeros((I.shape[0], 1))
    return numpy.hstack((P, I))


def add_padding_back(I):
    P = numpy.zeros((I.shape[0], 1))
    return numpy.hstack((I, P))


def soft2hard(probas):
    return probas == numpy.expand_dims(numpy.max(probas, axis=1), axis=1)


def probas2indices(probas):
    return numpy.argmax(probas, axis=1)


def hard2indices(probas):
    indices = list(numpy.nonzero(probas)[0])
    return indices


def hard2values(probas, index2value):
    indices = hard2indices(probas)
    values = set(indices2values(indices, index2value))
    return values


# def accuracy(Y_true, Y_pred):
# 	m = Y_true.shape[0]
# 	return float(numpy.sum(numpy.multiply(Y_true, Y_pred))) / m


def viterbi(P, A0, A):
    P = numpy.log(P)
    A0 = numpy.log(A0)
    A = numpy.log(A)

    n_steps, n_states = P.shape
    tags = range(n_states)
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for tag in tags:
        V[0][tag] = A0[tag] + P[0, tag]
        path[tag] = [tag]

    # Run Viterbi for t > 0
    for step in range(1, n_steps):
        V.append({})
        newpath = {}

        for tag in tags:
            (prob, state) = max((V[step - 1][tag0] + A[tag0, tag] + P[step, tag], tag0) for tag0 in tags)
            V[step][tag] = prob
            newpath[tag] = path[state] + [tag]

        # Don't need to remember the old paths
        path = newpath

    # Return the most likely sequence over the given time frame
    n = n_steps - 1
    (prob, state) = max((V[n][y], y) for y in tags)
    P_best = one_of_k(numpy.array([path[state]]), n_states)[0]
    return (numpy.exp(prob), path[state], P_best)


def extract_transitions(P):
    n_steps, n_states = P.shape
    A0 = numpy.zeros((n_states,))
    A = numpy.zeros((n_states, n_states))

    a_prev = P[0, :]
    A0 += a_prev

    for i in range(1, n_steps):
        a = P[i, :]
        A[a_prev == 1] += a
        a_prev = a

    return A0, A


def extract_windows(X, neighborhood_size=5, padding_token=0):
    N = X.shape[1]
    padding = numpy.ones((1, neighborhood_size)) * padding_token
    X_padded = numpy.hstack((padding, X, padding))

    X_batch = numpy.zeros((N, neighborhood_size * 2 + 1))
    for i in range(N):
        start = i
        end = start + neighborhood_size * 2 + 1
        X_batch[i, :] = X_padded[0, start: end]

    return N, X_batch


def extract_window_around_span(X, span, window_size, left_padding_token=0):
    left, right = span

    N = X.shape[0]
    if N < window_size:
        N_pad = window_size - N
        padding = numpy.int64(numpy.ones(N_pad) * left_padding_token)
        X_sub = numpy.hstack((padding, X))
        offset = N_pad
    else:
        center = (right + left) / 2
        window_size_l = (window_size + 1) / 2
        window_size_r = (window_size) / 2
        window_left = center - window_size_l
        window_right = center + window_size_r

        if window_left < 0:
            window_left = 0
        elif window_right > N:
            window_left = N - window_size

        window_right = window_left + window_size
        X_sub = X[window_left:window_right]
        offset = -window_left

    window_left = left + offset
    window_right = right + offset

    # D_pre = list(range(-window_left, 0, 1))
    # D_in = [0] * (window_right - window_left)
    # D_post = list(range(1, 1 + window_size - window_right))
    # D = numpy.array(D_pre + D_in + D_post)
    # # D += window_size - 1  # Shift the distances so that we only have positive numbers (indices)

    D = get_relative_distances_to_span(window_size, (window_left, window_right), window_size)
    return X_sub, D


def extract_window_around_span2(X, span, window_size, left_padding_token=0):
    left, right = span

    N = X.shape[0]
    if N < window_size:
        N_pad = window_size - N
        padding = numpy.array([left_padding_token] * N_pad)
        X_sub = numpy.hstack((padding, X))
        offset = N_pad
    else:
        center = (right + left) / 2
        window_size_l = (window_size + 1) / 2
        window_size_r = (window_size) / 2
        window_left = center - window_size_l
        window_right = center + window_size_r

        if window_left < 0:
            window_left = 0
        elif window_right > N:
            window_left = N - window_size

        window_right = window_left + window_size
        X_sub = X[window_left:window_right]
        offset = -window_left

    window_left = left + offset
    window_right = right + offset

    return X_sub, (window_left, window_right)


def get_relative_distances_to_span(N, span, max_distance):
    left, right = span

    D_pre = list(range(-left, 0, 1))
    D_in = [0] * (right - left)
    D_post = list(range(1, 1 + N - right))
    D = numpy.array(D_pre + D_in + D_post)
    D[D > max_distance] = max_distance
    D[D < -max_distance] = -max_distance
    D += max_distance  # Shift the distances so that we only have positive numbers (indices)
    return D


def tokens_to_binary_bag_of_words(tokens, tokens2indices):
    indices = values2indices(tokens, tokens2indices)
    bbow = numpy.zeros(len(tokens2indices))
    # for i in indices:
    #     bbow[i] = 1
    if len(indices) > 0:
        bbow[indices] = 1
    return bbow


def indices_to_binary_bag_of_words(indices, vocab_size):
    bbow = numpy.zeros(vocab_size)
    # for i in indices:
    #     bbow[i] = 1
    if len(indices) > 0:
        bbow[indices] = 1
    return bbow


class TFIDFStore:
    def __init__(self):
        self.tf = defaultdict(float)
        self.df = defaultdict(set)
        self.documents = set()
        self.terms = set()

    def add_term(self, document, term):
        self.tf[document, term] += 1.0
        self.df[term].add(document)
        self.documents.add(document)
        self.terms.add(term)

    def get_tf(self, document, term):
        return float(self.tf[document, term])

    def get_df(self, term):
        return float(len(self.df[term]))

    def get_tfidf(self, document, term):
        tf = self.get_tf(document, term)
        df = self.get_df(term)
        if df == 0:
            return 0
        idf = numpy.log(float(len(self.documents)) / df)
        tfidf = tf * idf
        return tfidf

    def print_tfidfs(self):
        with open("../res/tfidf_en.txt", "w") as f:
            m = max(map(len, self.terms)) + 1
            for t in self.terms:
                s = t + " " * (m - len(t))
                for d in self.documents:
                    s += "%f (%d) \t" % (self.get_tfidf(d, t), self.tf[d, t])
                print(s)
                f.write(s)

    def save(self, filepath):
        with io.open(filepath, "w", encoding="utf-8") as f:
            tf_line = json.dumps(self.tf)
            df_line = json.dumps(self.df)
            documents_line = json.dumps(self.documents)
            terms_line = json.dumps(self.terms)

            f.write(tf_line + u"\n")
            f.write(df_line + u"\n")
            f.write(documents_line + u"\n")
            f.write(terms_line + u"\n")

    def load(self, filepath):
        with io.open(filepath, encoding="utf-8") as f:
            tf_line = f.readline()
            df_line = f.readline()
            documents_line = f.readline()
            terms_line = f.readline()

            self.tf = json.loads(tf_line)
            self.df = json.loads(df_line)
            self.documents = json.loads(documents_line)
            self.terms = json.loads(terms_line)


def mark_unknown(tokens, token2index):
    prefix = u" <? "
    suffix = u" ?> "
    return [prefix + t.upper() + suffix if t not in token2index else t for t in tokens]


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def compute_class_weights(L, alpha=1.0):
    label_counts = Counter()
    label_counts.update(L)
    d = numpy.array(label_counts.values(), dtype="float")
    d = d ** alpha
    total_sum = numpy.sum(d)
    w = 1. - d / total_sum
    w = dict(zip(label_counts.keys(), w))
    return w
