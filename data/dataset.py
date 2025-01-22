from torch.utils.data import Dataset
import pandas as pd
import torch as torch
import torch.nn.functional as F


class LineTypeDataset(Dataset):

    def __init__(self, data_file_path):
        self.dataset = pd.read_csv(data_file_path)
        self.encoding_size = self.dataset['line'].str.len().max() # one-hot encode to largest string in dataset
        self.vocab = {c : i for i, c in enumerate(set(''.join(self.dataset['line'])))} # lookup on a character for one-hot index


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # Make a one-hot vector for each character in a string
        one_hot = F.one_hot(
            torch.tensor([self.vocab.get(c) for c in self.dataset.iloc[idx]['line']]),
            num_classes=len(self.vocab)
        )

        # Pad one-hot to largest string length
        return F.pad(
            one_hot,
            (0, 0, 0, self.encoding_size - len(one_hot)),
            mode='constant',
            value=0
        ).to(torch.float32), torch.tensor(self.dataset.iloc[idx]['type'], dtype=torch.float32)
