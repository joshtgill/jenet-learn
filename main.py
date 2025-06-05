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
import deploy.deployer as deployer


DATA_RES_PATH = 'data/res/'
MODEL_PATH = 'model/'
DATA_SOURCES = {
    'NUMBER': NumberAdapter(),
    'PRICE': PriceAdapter(),
    'DATE': DateAdapter(),
    'TIME': TimeAdapter(),
    'NAME': NameAdapter(DATA_RES_PATH + 'first_names.txt',
                        DATA_RES_PATH + 'last_names.txt'),
    'CITY': CityAdapter(DATA_RES_PATH + 'cities.txt'),
    'ADDRESS': AddressAdapter(DATA_RES_PATH + 'addresses.txt'),
    'ORDER_NUMBER': NumberAdapter(min_length=5,
                                  max_length=20,
                                  contain_digits=True,
                                  contain_letters=True,
                                  contains_decimal=False,
                                  prefix=''),
    'TRACKING_NUMBER': NumberAdapter(min_length=10,
                                     max_length=20,
                                     contain_digits=True,
                                     contain_letters=True,
                                     contains_decimal=False,
                                     prefix='T-'),
    'AIRPORT': TextAdapter(DATA_RES_PATH + 'airport_codes.txt',
                           DATA_RES_PATH + 'airport_names.txt'),
    'LANGUAGE': TextAdapter(DATA_RES_PATH + 'language.txt')
}
dataset = Dataset(DATA_RES_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make', type=int, metavar='<n>', help='make a dataset with n total samples')
    parser.add_argument('-t', '--train', action='store_true', help='train the model with the stored dataset')
    parser.add_argument('-q', '--query', type=str, metavar='<line>', help='query the model on a given line')
    parser.add_argument('-a', '--api', action='store_true', help='use the deployed model via the API')
    parser.add_argument('-d', '--deploy', action='store_true', help='deploy the model to the server')
    parser.add_argument('-b', '--batch_size', type=int, metavar='<batch size>', help='the batch size to train on', default=learner.DEFAULT_BATCH_SIZE)
    parser.add_argument('-e', '--num_epochs', type=int, metavar='<number of epochs>', help='the number of epochs to train over', default=learner.DEFAULT_BATCH_SIZE)

    print()
    args = parser.parse_args()
    if args.make:
        dataset.make(DATA_SOURCES, args.make)
        print()
    if args.train:
        learner.train("data/res/dataset.csv", args.batch_size, args.num_epochs)
        print()
    if args.query:
        print(f'{args.query} is of type {learner.query(args.query, args.api)}')
    if args.deploy:
        deployer.compress_model()
        deployer.upload_model()

    if not (args.make or args.train or args.query or args.deploy):
        parser.print_help()
