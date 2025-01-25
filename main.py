import argparse
from data_adapters.price_adapter import PriceAdapter
from data_adapters.date_adapter import DateAdapter
from data_adapters.name_adapter import NameAdapter
from dataset import Dataset as Dataset
import learner as learner
from line_vectorizer import LineVectorizer


RES_PATH = 'res/'
MODEL_FILE_PATH = RES_PATH + 'model.pt'
DATA_SOURCES = {
    'price': PriceAdapter(),
    'date': DateAdapter(),
    'name': NameAdapter(RES_PATH)
}
dataset = Dataset(RES_PATH, LineVectorizer)


def make(num_samples):
    dataset.make(DATA_SOURCES, num_samples)


def train():
    learner.learn(dataset, RES_PATH)


def query(line):
    pred = learner.query(RES_PATH, line)
    label = next(label for type, (label, _) in enumerate(DATA_SOURCES.items())
                 if type == pred)
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
