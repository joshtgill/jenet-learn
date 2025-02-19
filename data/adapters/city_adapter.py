from data.adapters.text_adapter import TextAdapter
import random


class CityAdapter(TextAdapter):

    def sample(self, k):
        samples = []
        for _ in range(k):
            city = random.choice(self.srcs[0])
            if random.randint(0, 1):
                city = city.replace(' ', '')
            if random.randint(0, 1):
                city = city.replace(',', '')
            if random.randint(0, 1):
                city = city.lower()
            elif random.randint(0, 1):
                city = city.upper()

            samples.append(city)

        return samples
