from gensim.models.base_any2vec import BaseWordEmbeddingsModel
import numpy as np
from preprocess import load_data

class vocab():
    def __init__(self):
        self.word_counts = {}
        self.word_index = {}
        self.index_word = {}
        self.n_grams = {}
        self.hashes = {}

    # Build vocabulary
    def build_vocab(self, sentences, min_n, max_n, bucket):

        id = 0
        for sentence in sentences:
            for word in sentence:
                if word not in self.word_counts:
                    self.word_counts[word] = 1
                    self.word_index[word] = id
                    self.index_word[id] = word
                    id += 1

                    text_ngrams, hashes = ngram_hashes(word, min_n, max_n, bucket)
                    self.n_grams[word] = text_ngrams

                    for ngram, hash in zip(text_ngrams, hashes):
                        self.hashes[ngram] = hash

                else:
                    self.word_counts[word] += 1

class FastText():

    def __init__(self, size=100, window=5, min_count=5, negative=5, min_n=3, max_n=6, bucket=2000000):

        self.min_n = min_n
        self.max_n = max_n
        self.bucket = bucket
        self.size = size
        self.window = window
        self.min_count = min_count
        self.negative = negative
        self.vocab = vocab()

    # Build vocabulary
    def build_vocab(self, sentences):
        self.vocab.build_vocab(sentences, self.min_n, self.max_n, self.bucket)

    # Train
    def train(self, sentences, total_examples, epochs, batch_words=10000):
        self.epochs = epochs
        self.batch_words = batch_words

        trained_word_count = 0
        raw_word_count = 0
        job_tally = 0

        for epoch in range(self.epochs):
            trained_word_count_epoch, raw_word_count_epoch, job_tally_epoch = self._train_epoch(
                data_iterable, cur_epoch=cur_epoch, total_examples=total_examples,
                total_words=total_words, queue_factor=queue_factor, report_delay=report_delay)

            job_batch, batch_size = [], 0

            for sentence_id, sentence in enumerate(sentences):
                data_length = len(sentence)

                if batch_size + data_length <= self.batch_words:
                    batch_size += data_length
                    job_batch.append(sentence)
                else:

                    job_batch, batch_size = [sentence], data_length

            trained_word_count += trained_word_count_epoch
            raw_word_count += raw_word_count_epoch
            job_tally += job_tally_epoch




        # Get word representation
        # def word_vec(self, word):
        #     if word in self.vocab:
        #         return super(FastTextKeyedVectors, self).word_vec(word, use_norm)
        #     elif self.bucket == 0:
        #         raise KeyError('cannot calculate vector for OOV word without ngrams')
        #     else:
        #         word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
        #         ngram_hashes = ngram_hashes(word, self.min_n, self.max_n, self.bucket)
        #
        #         for nh in ngram_hashes:
        #             word_vec += self.vectors_ngrams[nh]
        #         return word_vec / len(ngram_hashes)

# Return the ngrams for the given word
def compute_ngrams(word, min_n, max_n):
    word = '<' + word + '>'
    ngrams = []

    for n in range(min_n, min(max_n, len(word)) + 1):
        for pos in range(0, len(word)-n+1):
            ngram = word[pos:pos+n]
            ngrams.append(ngram)

    return ngrams

# Compute word hash
def compute_hash(string):
    # Supress warning
    old_settings = np.seterr(all='ignore')
    h = np.uint32(2166136261)
    for c in string:
        h = h ^ np.uint32(ord(c))
        h = h * np.uint32(16777619)
    np.seterr(**old_settings)
    return h

# Compute ngrams hash
def ngram_hashes(word, min_n, max_n, num_buckets):
    text_ngrams = compute_ngrams(word, min_n, max_n)
    hashes = [compute_hash(n) % num_buckets for n in text_ngrams]

    return text_ngrams, hashes



# print(ngram_hashes('ibis', min_n=3, max_n=6, num_buckets=2000000))


sentences = load_data('precision')
model = FastText()
model.build_vocab(sentences)

# dct = Dictionary(sentences)
ibis = 1