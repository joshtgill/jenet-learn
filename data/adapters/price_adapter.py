from data.adapters.data_adapter import DataAdapter
import random


class PriceAdapter(DataAdapter):

    def __init__(self, res_path):
        super().__init__(res_path)


    def sample_line(self):
        oom = random.randint(0, 9) # vary order of magnitude
        price = random.randint(10**oom, 10**(oom + 1)) / 100

        return '${:,.2f}'.format(price)
