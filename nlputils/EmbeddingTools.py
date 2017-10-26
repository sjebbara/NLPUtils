import numpy
import os
from gensim.models.word2vec import Word2Vec
from nlputils import LearningTools


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
