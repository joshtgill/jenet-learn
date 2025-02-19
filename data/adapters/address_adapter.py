from data.adapters.text_adapter import TextAdapter
import random


class AddressAdapter(TextAdapter):

    def sample(self, k):
        samples = []
        for _ in range(k):
            address = random.choice(self.srcs[0])
            if random.randint(0, 1):
                address = address.replace(', ', ' ')
            if random.randint(0, 1):
                address = address.lower()
            elif random.randint(0, 1):
                address = address.upper()

            samples.append(address)

        return samples
