

# Return the ngrams for the given word
def word2ngrams(word, min_n=3, max_n=6):
    word = '<' + word + '>'
    ngrams = [word]

    if len(word) > min_n + 2:
        for n in range(min_n, min(max_n, len(word)-2)):
            for pos in range(0,len(word)-n+1):
                subword = word[pos:pos+n]
                ngrams.append(subword)

    return ngrams

