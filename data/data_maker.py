from data.adapters.price_adapter import PriceAdapter
from data.adapters.date_adapter import DateAdapter
from data.adapters.name_adapter import NameAdapter
import pandas as pd


class DataMaker:

    DATA_SOURCES = {
        'price': (PriceAdapter, 0),
        'date': (DateAdapter, 1),
        'name': (NameAdapter, 2)
    }

    def __init__(self, res_path, dataset_file_name):
        self.res_path = res_path
        self.dataset_file_path = f'{res_path}{dataset_file_name}'


    def make(self, total_num_samples):
        # Use remainder to make exact number of total desired samples
        rem = total_num_samples % len(self.DATA_SOURCES)

        dataset = pd.DataFrame({'line': [], 'type': []})
        for _, (Adapter, type) in self.DATA_SOURCES.items():
            adapter = Adapter(self.res_path)
            num_samples = total_num_samples // len(self.DATA_SOURCES)
            if rem > 0:
                num_samples += 1
                rem -= 1

            dataset = pd.concat([
                dataset,
                pd.DataFrame({
                    'line': adapter.make(num_samples),
                    'type': [type] * num_samples
                })
            ])
        dataset.to_csv(self.dataset_file_path, index=False)

        print(f'created dataset with {len(dataset):,} samples')

        return dataset
