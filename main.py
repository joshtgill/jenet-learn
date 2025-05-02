import argparse
from data.adapters.number_adapter import NumberAdapter
from data.adapters.price_adapter import PriceAdapter
from data.adapters.date_adapter import DateAdapter
from data.adapters.name_adapter import NameAdapter
from data.adapters.time_adapter import TimeAdapter
from data.adapters.city_adapter import CityAdapter
from data.adapters.address_adapter import AddressAdapter
from data.adapters.text_adapter import TextAdapter
from dataset import Dataset as Dataset
import learner as learner
from model.line_vectorizer import LineVectorizer
from model.model import Model


DATA_RES_PATH = 'data/res/'
MODEL_PATH = 'model/'
DATA_SOURCES = {
    'number': NumberAdapter(),
    'price': PriceAdapter(),
    'date': DateAdapter(),
    'time': TimeAdapter(),
    'name': NameAdapter(DATA_RES_PATH + 'first_names.txt',
                        DATA_RES_PATH + 'last_names.txt'),
    'city': CityAdapter(DATA_RES_PATH + 'cities.txt'),
    'address': AddressAdapter(DATA_RES_PATH + 'addresses.txt'),
    'order number': NumberAdapter(min_length=5,
                                  max_length=20,
                                  contain_digits=True,
                                  contain_letters=True,
                                  prefix=''),
    'tracking number': NumberAdapter(min_length=10,
                                     max_length=20,
                                     contain_digits=True,
                                     contain_letters=True,
                                     prefix='T-'),
    'airport': TextAdapter(DATA_RES_PATH + 'airport_codes.txt',
                           DATA_RES_PATH + 'airport_names.txt'),
    'language': TextAdapter(DATA_RES_PATH + 'language.txt')
}
dataset = Dataset(DATA_RES_PATH)


def query(line):
    pred = Model.load(MODEL_PATH).query(line)
    label = next(label for type, (label, _) in enumerate(DATA_SOURCES.items())
                 if type == pred)
    print(f'{line} is of type {label}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make', type=int, metavar='<n>', help='make a dataset with n total samples')
    parser.add_argument('-t', '--train', action='store_true', help='train the model with the stored dataset')
    parser.add_argument('-q', '--query', type=str, metavar='<line>', help='query the stored model on a given line')
    parser.add_argument('-r', '--train-ratio', type=float, metavar='<train ratio>', help='the ratio of data to train on', default=0.80)
    parser.add_argument('-b', '--batches', type=int, metavar='<batch size>', help='the batch size to train on', default=64)
    parser.add_argument('-e', '--epochs', type=int, metavar='<number of epochs>', help='the number of epochs to train over', default=5)

    print()
    args = parser.parse_args()
    if args.make:
        dataset.make(DATA_SOURCES, args.make)
        print()
    if args.train:
        dataset.load(LineVectorizer)
        learner.learn(dataset, args.train_ratio, args.batches, args.epochs, MODEL_PATH)
        print()
    if args.query:
        query(args.query)
        print()

    if not (args.make or args.train or args.query):
        parser.print_help()
