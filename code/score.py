from argparse import ArgumentParser
from gensim.models import KeyedVectors
from scipy.stats import spearmanr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("resource_folder", nargs='?', default='../resources/', help="Resource folder path")
    parser.add_argument("gold_file", nargs='?', default='combined.tab', help="Name of the gold file to use")
    parser.add_argument("model_name", nargs='?', default='embeddings.vec', help="Name of the embedding file to use")

    return parser.parse_args()


def score(resource_folder='../resources/', gold_file='combined.tab', model_name='embeddings.vec', model=None,  debug=False):

    # Read gold file to a list
    word_pairs = []
    gold = []
    with open(resource_folder + gold_file) as f:
        next(f)
        for line in f:
            fields = line.split('\t')
            word_pairs.append([fields[0].lower(), fields[1].lower()])
            gold.append(float(fields[2]))

    # Load the model and get vocabulary
    if model is None:
        model = KeyedVectors.load_word2vec_format(resource_folder + model_name, binary=False)

    # Calculate cossine similarity
    sim = []
    for tuple in word_pairs:
        word1 = tuple[0]
        word2 = tuple[1]

        sim.append(model.wv.similarity(word1, word2))

    # Spearman correlation
    corr, _ = spearmanr(gold, sim)

    return corr


if __name__ == '__main__':
    args = parse_args()

    # Calculate correlation
    corr = score(resource_folder=args.resource_folder, gold_file=args.gold_file, model_name=args.model_name, debug=True)

    print("Final Score:\t", corr)