from data_adapters.data_adapter import DataAdapter
import random


class PriceAdapter(DataAdapter):

    def sample(self):
        oom = random.randint(0, 9) # vary order of magnitude
        price = random.randint(10**oom, 10**(oom + 1)) / 100

        return '${:,.2f}'.format(price)
