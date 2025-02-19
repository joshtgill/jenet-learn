from data.adapters.base_adapter import BaseAdapter
import random


class PriceAdapter(BaseAdapter):

    def sample(self, k):
        samples = []
        for _ in range(k):
            oom = random.randint(0, 9) # vary order of magnitude
            samples.append('${:,.2f}'.format(random.randint(10**oom, 10**(oom + 1)) / 100))

        return samples
