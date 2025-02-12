from data.adapters.base_adapter import BaseAdapter
import string
import random


class OrderNumberAdapter(BaseAdapter):

    def sample(self):
        return self.generate_random_string(random.randint(5, 20))


    def generate_random_string(self, length, contains_characters=True, contains_numbers=True):
        characters = ''
        if contains_characters:
            characters += string.ascii_letters
        if contains_numbers:
            characters += string.digits

        return ''.join(random.choice(characters) for _ in range(length)).upper()
