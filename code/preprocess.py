import xml.etree.ElementTree as etree
import pickle
import glob
import re
from argparse import ArgumentParser
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import FastText

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("dictionary_name", nargs='?', default='sentences', help="Name of the dictionary file to use")
    parser.add_argument("mapping_name", nargs='?', default='mapping', help="Name of the mapping file to use")

    return parser.parse_args()

# Save dictionary to file
def save(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

# Load dictionary from file
def load(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def extract_sentences(file_name='sentences'):

    # Get an iterable
    file = '../resources/dataset/eurosense.v1.0.high-' + file_name + '.xml'
    context = etree.iterparse(file)

    sentences = []

    for event, elem in iter(context):
        if elem.tag == "sentence":
            id = elem.attrib["id"]

            if int(id) % 1000 == 0 and int(id) != 0:
                print('\tSentences read: ', id)
                break

        elif elem.tag == "text" and elem.attrib["lang"] == "en":
            # print(elem.text)
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

    # Check if sentences exists
    if glob.glob(path + file_name + '_sentences' + '.pkl'):
        print('\nSentences found!')
        sentences = load(path + file_name + '_sentences')
    else:
        # Check if dictionary exists
        if glob.glob(path + file_name + '.pkl'):
            print('\nDictionary found!')
            sentences = load(path + file_name)
        else:
            print('\nDictionary not found!')
            print('\nBuilding dataset from file')
            sentences = extract_sentences(file_name)

    # Print sample sentences
    print('\nSample sentences')
    for i in sentences[0:10]:
        print(i)

    model = FastText(size=4, window=3, min_count=1, min_n=3, max_n=3)  # instantiate
    model.build_vocab(sentences=sentences)
    model.train(sentences=sentences, total_examples=len(sentences), epochs=10)  # train

    print(model)

    print("nights" in model.wv.vocab)
    print("night" in model.wv.vocab)
    print(model.similarity("night", "nights"))

    return sentences

if __name__ == '__main__':
    args = parse_args()

    _ = load_data(file_name=args.dictionary_name, path=args.resource_folder)