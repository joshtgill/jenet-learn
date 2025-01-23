from torch.utils.data import Dataset
import pandas as pd
from data.adapters.price_adapter import PriceAdapter
from data.adapters.date_adapter import DateAdapter
from data.adapters.name_adapter import NameAdapter
import torch as torch


class LineTypeDataset(Dataset):

    DATA_SOURCES = {
        'price': (PriceAdapter, 0),
        'date': (DateAdapter, 1),
        'name': (NameAdapter, 2)
    }

    def __init__(self, res_path, dataset_file_name, vectorizer):
        self.res_path = res_path
        self.dataset_file_name = dataset_file_name
        self.dataset = pd.read_csv(res_path + dataset_file_name)
        self.vectorizer = vectorizer
        self.encoding_size = self.dataset['line'].str.len().max() # one-hot encode to largest string in dataset

        # build lookup on a character for one-hot index
        self.vocab = {}
        for char in ''.join(self.dataset['line']):
            if char in self.vocab:
                continue

            self.vocab.update({char: len(self.vocab)})


    def make(self, total_num_samples):
        # Use remainder to make exact number of total desired samples
        rem = total_num_samples % len(self.DATA_SOURCES)

        self.dataset = pd.DataFrame({'line': [], 'type': []})
        for _, (Adapter, type) in self.DATA_SOURCES.items():
            adapter = Adapter(self.res_path)
            num_samples = total_num_samples // len(self.DATA_SOURCES)
            if rem > 0:
                num_samples += 1
                rem -= 1

            self.dataset = pd.concat([
                self.dataset,
                pd.DataFrame({
                    'line': adapter.make(num_samples),
                    'type': [type] * num_samples
                })
            ])
        self.dataset.to_csv(self.res_path + self.dataset_file_name, index=False)

        print(f'created dataset with {len(self.dataset):,} samples')


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # Make a one-hot vector for each character in a string
        return self.vectorizer(self.dataset.iloc[idx]['line'], self.vocab, self.encoding_size), \
               torch.tensor(self.dataset.iloc[idx]['type'], dtype=torch.float32)
