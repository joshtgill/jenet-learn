from torch.utils.data import Dataset as TorchDataset
import pandas as pd
import torch as torch


class Dataset(TorchDataset):

    DATASET_FILE_NAME = 'dataset.csv'

    def __init__(self, res_path, Vectorizer):
        self.dataset_file_path = res_path + self.DATASET_FILE_NAME
        self.dataset = pd.read_csv(self.dataset_file_path)

        # build vocab/character lookup
        vocab = {}
        for char in ''.join(self.dataset.iloc[:, 0]):
            if char in vocab:
                continue

            vocab.update({char: len(vocab)})

        self.vectorizer = Vectorizer(
            vocab,
            self.dataset.iloc[:, 0].str.len().max() # encode size to largest word in vocab
        )


    def make(self, data_sources, total_num_samples):
        # Use remainder to make exact number of total desired samples
        rem = total_num_samples % len(data_sources)

        self.dataset = pd.DataFrame()
        for type, (_, adapter) in enumerate(data_sources.items()):
            num_samples = total_num_samples // len(data_sources)
            if rem > 0:
                num_samples += 1
                rem -= 1

            self.dataset = pd.concat([
                self.dataset,
                pd.DataFrame([adapter.make(num_samples), [type] * num_samples]).T
            ])
        self.dataset.to_csv(self.dataset_file_path, header=False, index=False)

        print(f'created dataset with {len(self.dataset):,} samples')


    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        # Vectorizer a row
        return self.vectorizer(self.dataset.iloc[idx][: self.dataset.shape[1] - 1]), \
               torch.tensor(self.dataset.iloc[idx].iloc[-1], dtype=torch.float32)
