import argparse
from data.dataset import LineTypeDataset as Dataset
import learn.learner as learner
from learn.vectorizer import Vectorizer
import torch


RES_PATH = 'data/res/'
DATASET_FILE_NAME = 'dataset.csv'
MODEL_FILE_PATH = RES_PATH + 'model.pt'
dataset = Dataset(RES_PATH, DATASET_FILE_NAME, Vectorizer())


def make(num_samples):
    dataset.make(num_samples)


def train():
    model = learner.learn(dataset)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': dataset.vocab,
        'encoding_size': dataset.encoding_size
    }, MODEL_FILE_PATH)


def query(line):
    pred = learner.query(MODEL_FILE_PATH, line)
    label = next(label for label, values in Dataset.DATA_SOURCES.items()
                 if values[1] == pred)
    print(f'{line} is of type {label}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make', type=int, metavar='n', help='make a dataset with n total samples')
    parser.add_argument('-t', '--train', action='store_true', help='train the model with the stored dataset')
    parser.add_argument('-q', '--query', type=str, metavar='line', help='query the stored model on a given line')

    print()
    args = parser.parse_args()
    if args.make:
        make(args.make)
        print()
    if args.train:
        train()
        print()
    if args.query:
        query(args.query)
        print()
