from preprocess import load_data
from argparse import ArgumentParser
from gridSearch import *

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("file_name", nargs='?', default='precision', help="Name of the dictionary file to use")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load sentences
    sentences = load_data(file_name=args.file_name)

    # Define the grid search parameters
    epochs = [20]
    negative = [5, 10]
    window = [3, 5, 7]
    embedding_size = [300]
    min_count = [5]
    min_n = [2, 3]
    max_n = [5, 6]
    param_grid = dict(epochs=epochs,
                      negative=negative,
                      window=window,
                      embedding_size=embedding_size,
                      min_count=min_count,
                      min_n=min_n,
                      max_n=max_n)

    # Train
    grid = gridSearch(sentences=sentences, param_grid=param_grid)
    grid.fit()

    # Print grid search summary
    grid.summary()