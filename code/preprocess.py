import xml.etree.ElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser
import numpy as np
# import fastText

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("file_name", nargs='?', default='precision', help="Name of the dictionary file to use")
    return parser.parse_args()

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def extract_sentences(file_name):

    # Get an iterable
    file = '../resources/dataset/eurosense.v1.0.high-' + file_name + '.xml'
    context = etree.iterparse(file)

    sentences = []

    for event, elem in iter(context):
        if elem.tag == "sentence":
            id = elem.attrib["id"]

            if int(id) % 10000 == 0 and int(id) != 0:
                print('\tSentences read: ', id)

        elif elem.tag == "text" and elem.attrib["lang"] == "en":
            if elem.text:
                sentence = re.sub("[^a-zA-Z]+", " ", elem.text).lower().split()
                sentences.append(sentence)

        elem.clear()

    save(sentences, '../resources/' + file_name)
    return sentences

# Convert to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    temp[data_point_index] = 1
    return temp


def load_data(file_name, path="../resources/"):

    # Check if dictionary exists
    if glob.glob(path + file_name + '.pkl'):
        print('\nSentences found!')
        sentences = load(path + file_name)
    else:
        print('\nSentences not found!')
        print('\nBuilding sentences from file')
        sentences = extract_sentences(file_name)

    # Print sample sentences
    print('\nSample sentences')
    for i in sentences[0:10]:
        print(i)

    # Show the ngrams of certain words
    # print('\nN-grams of a few words')
    # ids = np.random.randint(0,len(sentences), size=10)
    # for id in ids:
    #     pos = np.random.randint(0, len(sentences[id]))
    #     ngrams = fastText.word2ngrams(sentences[id][pos])
    #     print(sentences[id][pos] + ':', ' '.join(ngrams))

    return sentences

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(file_name=args.file_name, path=args.resource_folder)