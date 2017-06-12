import numpy
import LearningTools


class BatchIterator:
    def __init__(self, named_batch_iterable):
        self.iterable = named_batch_iterable

    def __iter__(self):
        for batches in self.iterable:
            try_next = True
            i = 0
            while try_next:
                instance = LearningTools.BetterDict()
                for name, batch in batches.iteritems():
                    if i < len(batch):
                        instance[name] = batch[i]
                    else:
                        try_next = False
                        break
                        # has_next = has_next and i < len(batch) - 1
                i += 1
                if try_next:
                    yield instance


class BatchGenerator:
    RAW_DATA_BATCH_NAME = "raw_data"

    def __init__(self, iterable, batch_size=1, vectorizer=None, raw_data_name=None, return_raw_data_batch=False):
        self.iterable = iterable
        self.batch_size = batch_size
        self.vectorizer = vectorizer
        self.return_raw_data_batch = return_raw_data_batch
        self.raw_data_name = raw_data_name

    def __iter__(self):
        batch = []
        for element in self.iterable:
            batch.append(element)

            current_batch_size = len(batch)
            if current_batch_size >= self.batch_size:
                yield self._vectorize_batch(batch)
                batch = []

        if len(batch) > 0:
            yield self._vectorize_batch(batch)

    def _vectorize_batch(self, batch):
        vectorized_batch = self.vectorizer.transform(batch)
        if self.return_raw_data_batch or self.raw_data_name:
            if type(vectorized_batch) is LearningTools.BetterDict:
                if self.raw_data_name:
                    vectorized_batch[self.raw_data_name] = batch
                else:
                    vectorized_batch[BatchGenerator.RAW_DATA_BATCH_NAME] = batch
                return vectorized_batch
            else:
                return batch, vectorized_batch

        return vectorized_batch


class Vectorizer(object):
    def __init__(self, name=None):
        self.name = name
        if self.name is None:
            self.name = type(self).__name__
        self.previous_vectorizer = None

    def _transform_instance(self, x):
        return x

    def _transform_batch(self, X):
        T = [self._transform_instance(x) for x in X]
        return T

    def transform(self, X):
        if self.previous_vectorizer:
            X = self.previous_vectorizer.transform(X)
        T = self._transform_batch(X)
        return T

    def __call__(self, previous_vectorizer):
        assert isinstance(previous_vectorizer, Vectorizer)
        if self.previous_vectorizer:
            print "Warning: Override previously assigned vectorizer:", self.previous_vectorizer.name
        self.previous_vectorizer = previous_vectorizer
        return self


class VectorizerUnion(Vectorizer):
    def __init__(self, previous_vectorizers=None, **kwargs):
        super(VectorizerUnion, self).__init__(**kwargs)

        if previous_vectorizers:
            self.previous_vectorizers = dict(previous_vectorizers)
        else:
            self.previous_vectorizers = dict()

    def transform(self, X):
        T = LearningTools.BetterDict()
        for name, vectorizer in self.previous_vectorizers.iteritems():
            T[name] = vectorizer.transform(X)
        return T

    def __call__(self, name, previous_vectorizer):
        assert isinstance(previous_vectorizer, Vectorizer)
        if name in self.previous_vectorizers:
            print "Warning: Override previously assigned vectorizer:", name, self.previous_vectorizers[name].name
        self.previous_vectorizers[name] = previous_vectorizer
        return self


class ZipVectorizer(Vectorizer):
    def __init__(self, previous_vectorizers=None, **kwargs):
        super(ZipVectorizer, self).__init__(**kwargs)

        if previous_vectorizers:
            self.previous_vectorizers = previous_vectorizers
        else:
            self.previous_vectorizers = []

    def transform(self, X):
        T = [v.transform(X) for v in self.previous_vectorizers]
        return zip(*T)

    def __call__(self, previous_vectorizers):
        if len(self.previous_vectorizers) > 0:
            print "Warning: Override previously assigned vectorizer:", previous_vectorizers
        self.previous_vectorizers = previous_vectorizers
        return self


class LambdaVectorizer(Vectorizer):
    def __init__(self, instance_function, **kwargs):
        super(LambdaVectorizer, self).__init__(**kwargs)
        self.instance_function = instance_function

    def _transform_instance(self, x):
        if self.instance_function:
            return self.instance_function(x)
        else:
            return x


class IteratorVectorizer(Vectorizer):
    def __init__(self, vectorizer, **kwargs):
        super(IteratorVectorizer, self).__init__(**kwargs)
        assert isinstance(vectorizer, Vectorizer)
        self.vectorizer = vectorizer

    def _transform_instance(self, iterable):
        T = []
        for x in iterable:
            T.append(self.vectorizer._transform_instance(x))
        return T


# class VectorizerSequence(Vectorizer):
#     def __init__(self, vectorizers, **kwargs):
#         super(VectorizerSequence, self).__init__(**kwargs)
#         self.vectorizers = vectorizers
#
#     def _transform_batch(self, X):
#         T = X
#         for v in self.vectorizers:
#             T = v._transform_batch(T)
#         return T
#
# class VectorizerUnion(Vectorizer):
#     def __init__(self, named_vectorizers, **kwargs):
#         super(VectorizerUnion, self).__init__(**kwargs)
#         self.named_vectorizers = named_vectorizers
#
#     def _transform_batch(self, X):
#         T = LearningTools.BetterDict()
#         for name, vectorizer in self.named_vectorizers:
#             T[name] = vectorizer._transform_batch(X)
#         return T


class ArrayVectorizer(Vectorizer):
    def _transform_batch(self, X):
        return numpy.array(X)


class Padded1DSequenceVectorizer(Vectorizer):
    def __init__(self, vocabulary, padding_position, **kwargs):
        super(Padded1DSequenceVectorizer, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.vectorizer = VocabSequenceVectorizer1D(vocabulary, mode="sequence")
        self.padder = VocabularyPaddingVectorizer(vocabulary, padding_position)

    def _transform_batch(self, sequences):
        vectorized_sequences = [self.vectorizer._transform_instance(sequence) for sequence in sequences]

        padded_sequences = self.padder._transform_batch(vectorized_sequences)
        return padded_sequences


class Padded2DSequenceVectorizer(Vectorizer):
    def __init__(self, vocabulary, padding_position, **kwargs):
        super(Padded2DSequenceVectorizer, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.vectorizer = VocabSequenceVectorizer2D(vocabulary, mode="sequence")
        self.padder = VocabularyPaddingVectorizer(vocabulary, padding_position)

    def _transform_batch(self, sequences):
        vectorized_sequences = [self.vectorizer._transform_instance(sequence) for sequence in sequences]

        padded_sequences = self.padder._transform_batch(vectorized_sequences)
        return padded_sequences


class VocabSequenceVectorizer1D(Vectorizer):
    def __init__(self, vocabulary, mode="sequence", **kwargs):
        super(VocabSequenceVectorizer1D, self).__init__(**kwargs)
        self.vocabulary = vocabulary
        self.mode = mode

    def _transform_instance(self, tokens):
        if self.mode == "sequence":
            t = list(self.vocabulary.get_indices(tokens))
        elif self.mode == "bow":
            t = list(self.vocabulary.to_bow(tokens))
        elif self.mode == "lbow":
            v = self.vocabulary.to_bow(tokens)
            v = numpy.log(v + 1)
            t = list(v)
        elif self.mode == "lnbow":
            v = self.vocabulary.to_bow(tokens)
            v = numpy.log(v + 1)
            v /= numpy.max(v)
            t = list(v)
        elif self.mode == "bbow":
            t = list(self.vocabulary.to_bbow(tokens))
        elif self.mode == "onehot":
            t = list(self.vocabulary.to_one_hot_sequence(tokens))
        else:
            raise ValueError("mode {} undefined".format(self.mode))
        return t


class VocabSequenceVectorizer2D(IteratorVectorizer):
    def __init__(self, vocabulary, mode="sequence"):
        super(VocabSequenceVectorizer2D, self).__init__(VocabSequenceVectorizer1D(vocabulary, mode))


class DocumentLevelTokenExtractor(Vectorizer):
    def _transform_instance(self, d):
        return d.tokens


class Opinions2TagSequence(Vectorizer):
    def __init__(self, tagging_scheme, **kwargs):
        super(Opinions2TagSequence, self).__init__(**kwargs)
        self.tagging_scheme = tagging_scheme

    def _transform_instance(self, d):
        spans = set([(o.token_start, o.token_end) for o in d.opinions])
        tags = self.tagging_scheme.spans2tags(len(d.tokens), spans)
        return tags


class TagSequenceVectorizer(Vectorizer):
    def __init__(self, tagging_scheme, **kwargs):
        super(TagSequenceVectorizer, self).__init__(**kwargs)
        self.tagging_scheme = tagging_scheme

    def _transform_instance(self, tags):
        return self.tagging_scheme.tags2encoding(tags)


class DocumentLevelPosTagExtractor(Vectorizer):
    def _transform_instance(self, d):
        return d.token_pos_tags


class SentenceLevelExtractor(Vectorizer):
    def __init__(self, sentence_extractor, **kwargs):
        super(SentenceLevelExtractor, self).__init__(**kwargs)
        assert isinstance(sentence_extractor, Vectorizer)
        self.sentence_extractor = sentence_extractor

    def _transform_instance(self, d):
        T = []
        for sentence in d.sentences:
            T.append(self.sentence_extractor._transform_instance(sentence))
        return T


class SentenceLevelTokenExtractor(SentenceLevelExtractor):
    def __init__(self, ):
        super(SentenceLevelTokenExtractor, self).__init__(DocumentLevelTokenExtractor())


class SentenceLevelPosTagExtractor(SentenceLevelExtractor):
    def __init__(self, ):
        super(SentenceLevelPosTagExtractor, self).__init__(DocumentLevelPosTagExtractor())


class PaddingVectorizer(Vectorizer):
    def __init__(self, padding_value, padding_position, **kwargs):
        super(PaddingVectorizer, self).__init__(**kwargs)
        self.padding_value = padding_value
        self.padding_position = padding_position

    def _transform_batch(self, X):
        X_padded = LearningTools.pad(X, self.padding_position, self.padding_value)
        X_padded = X_padded.astype(int)
        return X_padded


class PaddingVectorizer1D(Vectorizer):
    def __init__(self, padding_value, padding_position, **kwargs):
        super(PaddingVectorizer1D, self).__init__(**kwargs)
        self.padding_value = padding_value
        self.padding_position = padding_position

    def _transform_batch(self, X):
        max_len = max(len(x) for x in X)

        if self.padding_position == "pre":
            X_padded = [[self.padding_value] * (max_len - len(x)) + x for x in X]
        elif self.padding_position == "post":
            X_padded = [x + [self.padding_value] * (max_len - len(x)) for x in X]
        return X_padded


class VocabularyPaddingVectorizer(PaddingVectorizer):
    def __init__(self, vocabulary, padding_position):
        super(VocabularyPaddingVectorizer, self).__init__(vocabulary.padding_index, padding_position)
