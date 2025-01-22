import argparse
from data.data_maker import DataMaker
from data.dataset import LineTypeDataset
import learn as learner


RES_PATH = 'data/res/'
LINE_TYPE_DATASET_FILE_NAME = 'line_type_dataset.csv'


def make(num_samples):
    DataMaker(RES_PATH, LINE_TYPE_DATASET_FILE_NAME).make(num_samples)


def train():
    learner.learn(LineTypeDataset(RES_PATH + LINE_TYPE_DATASET_FILE_NAME))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--make', type=int, metavar='n', help='make a dataset with n total samples')
    parser.add_argument('-t', '--train', action='store_true', help='train the model with the stored dataset')

    print()
    args = parser.parse_args()
    if args.make:
        make(args.make)
        print()
    if args.train:
        train()
        print()
