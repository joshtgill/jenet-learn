from data.adapters.data_adapter import DataAdapter
import random


class NameAdapter(DataAdapter):

    def __init__(self, res_path):
        super().__init__(res_path)

        with open(f'{self.res_path}/first_names.txt', 'r') as file:
            self.first_names = [line.strip() for line in file.readlines()]
        with open(f'{self.res_path}/last_names.txt', 'r') as file:
            self.last_names = [line.strip() for line in file.readlines()]


    def sample_line(self):
        return f'{random.choice(self.first_names)} {random.choice(self.last_names)}'
