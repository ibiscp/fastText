from fnvhash import fnv1a_32
import numpy as np

# Return the ngrams for the given word
def word2ngrams(word, min_n=3, max_n=3):
    word = '<' + word + '>'
    ngrams = [word]

    if len(word) > min_n + 1:
        for n in range(min_n, max_n+1):
            for pos in range(0,len(word)-n+1):
                subword = word[pos:pos+n]
                ngrams.append(subword)

    return ngrams

print(fnv1a_32(b'foo'))
print(fnv1a_32(b'ibis'))
print(fnv1a_32(b'prevedello'))

dt1=np.dtype(('i4', [('bytes','u1',4)]))

print(dt1)

print(np.int8('ibis'))

# # Get vocabulary
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(sentences)
# V = len(tokenizer.word_index) + 1
# vocab = tokenizer.word_index.keys()
#
# ngrams2Idx = {}
# ngrams_vocab = []
# vocab_ngrams = {}
# for i in vocab:
#     ngrams_vocab += word2ngrams(i)
#     vocab_ngrams[i] = word2ngrams(i)
#
# ngrams2Idx = dict((c, i + len(vocab)) for i, c in enumerate(ngrams_vocab))
# ngrams2Idx.update(tokenizer.word_index)
# words_and_ngrams_vocab = len(ngrams2Idx)
#
#
#
#
# # Collect words
# n_grams_dic = {}
# words = []
# for sentence in sentences:
#     for word in sentence:
#         words.append(word)
#         # ngrams = word2ngrams(word)
#         # if word not in n_grams_dic:
#         #     n_grams_dic[word] = {'subwords':ngrams, 'count':1}
#         #     words += ngrams
#         # else:
#         #     n_grams_dic[word]['count'] = n_grams_dic[word]['count'] + 1
# words = set(words)
#
# word2int = {}
# int2word = {}
#
# vocab_size = len(words)
#
# for i, word in enumerate(words):
#     word2int[word] = i
#     int2word[i] = word
#
# x = []
# y = []
# WINDOW_SIZE = 2
# for sentence in sentences:
#     for word_index, word in enumerate(sentence):
#         word_grams = n_grams_dic[word]
#         for nb_word in sentence[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(sentence)) + 1]:
#             nb_word_grams = n_grams_dic[nb_word]
#             if nb_word != word:
#                 for w in word_grams:
#                     for nb in nb_word_grams:
#                         x.append(w)
#                         y.append(nb)
#
# # Print sample ngrams
# print('\nSample ngrams')
# for i in range(0,5):
#     print(x[i], y[i])

# t = Tokenizer(num_words=30000, oov_token='<OOV>')
# t.fit_on_texts(x)
# t.fit_on_texts(y)
# # save(t, path + 'tokenizer')
#
# # Apply tokenizer
# train_x = t.texts_to_sequences(train_words_x)
# train_y = t.texts_to_sequences(train_words_y)

# x_train = []  # input word
# y_train = []  # output word
# for data_word in data:
#     x_train.append(to_one_hot(word2int[data_word[0]], vocab_size))
#     y_train.append(to_one_hot(word2int[data_word[1]], vocab_size))
# # convert them to numpy arrays
# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
#
#
# # Save sentences
# save(sentences, path + dictionary_name + '_sentences')