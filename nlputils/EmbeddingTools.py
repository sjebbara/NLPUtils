import numpy
import os
from gensim.models.word2vec import Word2Vec
from nlputils import LexicalTools

from nlputils import LearningTools
from nlputils import DataTools


def export_w2v(output_dirname, name, w2v=None, input_filename=None):
    if w2v is None:
        w2v = Word2Vec.load(input_filename)

    assert isinstance(w2v, Word2Vec)
    if not os.path.isdir(output_dirname):
        os.mkdir(output_dirname)

    w2v.save_word2vec_format(os.path.join(output_dirname, name + "_embeddings.bin.w2v"),
                             os.path.join(output_dirname, name + "_vocab.w2v"), binary=True)


def load(filepath, name):
    w2v = Word2Vec.load_word2vec_format(os.path.join(filepath, name + "_embeddings.bin.w2v"),
                                        os.path.join(filepath, name + "_vocab.w2v"), binary=True)
    return w2v


def load_from_gensim(filepath):
    w2v = Word2Vec.load(filepath)
    return w2v


def load_remapped(filepath, name, keymap_filepath):
    def to_key_value(line):
        parts = line.split()
        key = parts[0]
        value = parts[1]
        return key, value

    w2v = load(filepath, name)
    key2uri = LearningTools.load_as_dict(keymap_filepath, to_key_value=to_key_value)

    remapped_vocab = dict([(key2uri[k], v) for k, v in w2v.vocab.iteritems()])
    remapped_index2word = [key2uri[k] for k in w2v.index2word]

    w2v.vocab = remapped_vocab
    w2v.index2word = remapped_index2word
    return w2v


def get_phrase_vector(w2v, phrase):
    v = 0
    for w in phrase:
        v += w2v[w]

    return v


def similarity_of_phrases(w2v, phrase1, phrase2):
    v1 = get_phrase_vector(w2v, phrase1)
    v2 = get_phrase_vector(w2v, phrase2)

    return cosine_similarity(v1, v2)


def cosine_similarity(v1, v2):
    return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))


def _mse_distance(W, x):
    return numpy.mean((x - W) ** 2, axis=1)


def _cosine_distance(W, x):
    W = W / numpy.expand_dims(numpy.linalg.norm(W, axis=1), axis=-1)
    x = x / numpy.linalg.norm(x)
    return -numpy.dot(W, x)


def get_nearest_neighbors(W, x, top_k, index2word):
    D = _mse_distance(W, x)
    S = numpy.argsort(D)
    Iknn = S[1:top_k + 1]
    words_knn = [index2word[i] for i in Iknn]
    return zip(words_knn, W[Iknn, :])


def compose_ngram_word_embedding(ngrams, char_ngram_embeddings, n_slots=3, sigma=0.2, normalize=False,
                                 drop_unknown=True):
    def _kernel(x, y):
        return numpy.exp(-(x - y) ** 2 / (2 * sigma ** 2))

    def _weights(n_ngrams, n_slots):
        weights = numpy.zeros((n_ngrams, n_slots))
        for i_slot in range(n_slots):
            for i_ngram in range(n_ngrams):
                r_slot = float(i_slot) / (n_slots - 1)
                r_ngram = float(i_ngram) / (n_ngrams - 1)

                weights[i_ngram, i_slot] = _kernel(r_slot, r_ngram)

        return weights

    ngram_vector_size = char_ngram_embeddings.W.shape[1]

    n_ngrams = len(ngrams)
    W = _weights(n_ngrams, n_slots)

    if normalize:
        W = W / numpy.sum(W, axis=0, keepdims=True)

    vector_parts = []
    for i_slot in range(n_slots):
        v_slot = numpy.zeros(ngram_vector_size)
        for i_ngram in range(n_ngrams):
            ngram = ngrams[i_ngram]
            if ngram in char_ngram_embeddings:
                ngram_vector = char_ngram_embeddings.get_vector(ngram)

                v_slot += W[i_ngram, i_slot] * ngram_vector
            elif drop_unknown:
                # do nothing if unknown (same as add "zeros")
                pass
            else:
                raise ValueError("unknown ngram: " + ngram)

        vector_parts.append(v_slot)
    vector = numpy.concatenate(vector_parts)
    # if normalize:
    #     vector /= numpy.linalg.norm(vector)

    return vector


def compose_ngram_word_embeddings(words, ngram_size, char_ngram_embeddings, n_slots=3, sigma=0.2, normalize=False,
                                  pad_start=None, pad_end=None, drop_unknown=True):
    words = list(words)

    W = []
    for word in words:
        char_ngrams = LexicalTools.get_n_grams(list(word), ngram_size, pad_start=pad_start, pad_end=pad_end)
        char_ngrams = ["".join(ngram) for ngram in char_ngrams]

        vector = compose_ngram_word_embedding(char_ngrams, char_ngram_embeddings, n_slots=n_slots, sigma=sigma,
                                              normalize=normalize, drop_unknown=drop_unknown)
        W.append(vector)

    W = numpy.array(W)
    word_vocabulary = DataTools.Vocabulary()
    word_vocabulary.init_from_word_list(words)

    char_ngram_word_embeddings = DataTools.Embedding()
    char_ngram_word_embeddings.init(word_vocabulary, W)
    return char_ngram_word_embeddings


def concatenate_embeddings(embeddings_list, drop_unknown=False):
    words = None
    for embeddings in embeddings_list:
        if words is None:
            words = set(embeddings.vocabulary.index2word)
        else:
            if drop_unknown:
                words = words & set(embeddings.vocabulary.index2word)
            else:
                words = words | set(embeddings.vocabulary.index2word)
    W = []
    index2word = list(words)

    for i, word in enumerate(index2word):
        vectors = []
        for embeddings in embeddings_list:
            vectors.append(embeddings.get_vector(word))

        vector = numpy.concatenate(vectors)
        W.append(vector)
    W = numpy.array(W)

    new_vocabulary = DataTools.Vocabulary()
    new_vocabulary.init_from_word_list(index2word)
    new_embeddings = DataTools.Embedding()
    new_embeddings.init(new_vocabulary, W)
    return new_embeddings
