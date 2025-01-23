from torch.utils.data import Dataset
import pandas as pd
import torch as torch


class LineTypeDataset(Dataset):

    def __init__(self, data_file_path, vectorizer):
        self.dataset = pd.read_csv(data_file_path)
        self.vectorizer = vectorizer
        self.encoding_size = self.dataset['line'].str.len().max() # one-hot encode to largest string in dataset

        # build lookup on a character for one-hot index
        self.vocab = {}
        for char in ''.join(self.dataset['line']):
            if char in self.vocab:
                continue

            self.vocab.update({char: len(self.vocab)})


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # Make a one-hot vector for each character in a string
        return self.vectorizer(self.dataset.iloc[idx]['line'], self.vocab, self.encoding_size), \
               torch.tensor(self.dataset.iloc[idx]['type'], dtype=torch.float32)
