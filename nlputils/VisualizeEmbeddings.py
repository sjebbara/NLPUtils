import numpy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from nlputils import DataTools
from nlputils import LearningTools

__author__ = 'sjebbara'


def distance(W, x):
    return numpy.sqrt(numpy.sum((x - W) ** 2, axis=1))


def cosine_distance(W, x):
    W = W / numpy.expand_dims(numpy.linalg.norm(W, axis=1), axis=-1)
    x = x / numpy.linalg.norm(x)
    return -numpy.dot(W, x)


def visualize_most_common_embeddings(embedding, top_k):
    visualize_embeddings(embedding.W, embedding.vocabulary.most_common(top_k), embedding.vocabulary.word2index)


def visualize_embeddings(W, words, word2index, labels=None):
    I = DataTools.values2indices(words, word2index)
    X = numpy.array(W[I], dtype="float64")

    tsne = TSNE(n_components=2)
    E = tsne.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(E[:, 0], E[:, 1], s=4)

    if labels is None:
        labels = words

    for txt, vec in zip(labels, E):
        # try:
        #     # print(txt)
        #     utxt = txt.decode("utf-8")
        # except UnicodeEncodeError as e:
        #     utxt = txt
        #     print(txt, ":")
        #     print(e)
        # if txt.endswith("en"):
        ax.text(vec[0], vec[1], txt, fontsize=14, color="b")
    # else:
    #     ax.text(vec[0], vec[1], utxt, fontsize=10, color="g")

    plt.show()


def visualize_bilingual_embeddings(W_l1, W_l2, words_l1, words_l2, word2index_l1, word2index_l2):
    I_l1 = DataTools.values2indices(words_l1, word2index_l1)
    I_l2 = DataTools.values2indices(words_l2, word2index_l2)

    X_l1 = numpy.array(W_l1[I_l1], dtype="float64")
    X_l2 = numpy.array(W_l2[I_l2], dtype="float64")

    X = numpy.concatenate((X_l1, X_l2), axis=0)

    tsne = TSNE(n_components=2)
    E = tsne.fit_transform(X)

    fig, ax = plt.subplots()
    ax.scatter(E[:, 0], E[:, 1], s=4)

    for i, txt in enumerate(words_l1):
        vec = E[i, :]
        try:
            utxt = txt.decode("utf-8")
        except UnicodeEncodeError as e:
            utxt = txt
            print(txt, ":")
            print(e)
        ax.text(vec[0], vec[1], utxt, fontsize=10, color="g")

    for i, txt in enumerate(words_l2):
        vec = E[i + len(words_l1), :]
        try:
            utxt = txt.decode("utf-8")
        except UnicodeEncodeError as e:
            utxt = txt
            print(txt, ":")
            print(e)
        ax.text(vec[0], vec[1], utxt, fontsize=10, color="b")

    plt.show()


def print_analysis(W, words, top_k, word2index, index2word, f=None):
    print_crosslingual_analysis(W, W, words, top_k, word2index, index2word, f)


def print_crosslingual_analysis(W_l1, W_l2, words_l1, top_k, word2index_l1, index2word_l2, f=None):
    for word_l1 in words_l1:
        LearningTools.log(f, "###########")
        LearningTools.log(f, word_l1)
        LearningTools.log(f, "-----------")
        try:
            x = W_l1[word2index_l1[word_l1], :]
            # x = numpy.expand_dims(x, axis=0)
            # D = distance(W_l2, x)
            D = cosine_distance(W_l2, x)
            S = numpy.argsort(D)
            Iknn = S[1:top_k + 1]
            print(Iknn)
            words_knn_l2 = [index2word_l2[i] for i in Iknn]
            for i, word_knn_l2 in enumerate(words_knn_l2):
                LearningTools.log(f, "%s.) %s\t%s" % (i, D[S[i + 1]], word_knn_l2))
        except KeyError as e:
            LearningTools.log(f, "No entry for token %s" % word_l1)
            LearningTools.log(f, e)


def get_nearest_neighbors(W, x, top_k, index2word):
    # x = numpy.expand_dims(x, axis=0)
    D = cosine_distance(W, x)
    S = numpy.argsort(D)
    Iknn = S[1:top_k + 1]
    words_knn = [index2word[i] for i in Iknn]
    return zip(words_knn, W[Iknn, :])


def analogy(W, index2word, word2index, entity1, property1, entity2=None, property2=None, top_k=5):
    if property2:
        entity2 = W[word2index[entity1]] - W[word2index[property1]] + W[word2index[property2]]
        knn = get_nearest_neighbors(W, entity2, top_k, index2word)
    elif entity2:
        property2 = W[word2index[property1]] - W[word2index[entity1]] + W[word2index[entity2]]
        knn = get_nearest_neighbors(W, property2, top_k, index2word)

    return zip(*knn)[0]


def knn(W, index2word, word2index, word, top_k=5):
    x = W[word2index[word]]
    knn = get_nearest_neighbors(W, x, top_k, index2word)
    return zip(*knn)[0]
