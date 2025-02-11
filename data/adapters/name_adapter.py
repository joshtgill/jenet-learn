from data.adapters.base_adapter import BaseAdapter
import random


class NameAdapter(BaseAdapter):

    FORMATS = [
        '{0}',          # Josh
        '{0} {1}',      # Josh Gillette
        '{0}{1}',       # JoshGillette
        '{0} {1:.1}',   # Josh G
        '{0} {1:.1}.',  # Josh G.
        '{0}{1:.1}',    # JoshG
        '{0}{1:.1}.',   # JoshG.
        '{1} {0}',      # Gillette Josh
        '{1},{0}',      # Gillette, Josh
        '{1}{0}',       # GilletteJosh
        '{0:.1} {1}',   # J Gillette
        '{0:.1}. {1}',  # J. Gillette
        '{0:.1}{1}',    # JGillette
        '{0:.1}.{1}',   # J.Gillette
        '{1}, {0:.1}',  # Gillette, J
        '{1}, {0:.1}.', # Gillette, J.
        '{1},{0:.1}',   # Gillette,J
        '{1},{0:.1}.'   # Gillette,J.
    ]

    def __init__(self, data_res_path):
        with open(f'{data_res_path}/first_names.txt', 'r') as file:
            self.first_names = [line.strip() for line in file.readlines()]
        with open(f'{data_res_path}/last_names.txt', 'r') as file:
            self.last_names = [line.strip() for line in file.readlines()]


    def sample(self):
        return random.choice(self.FORMATS).format(random.choice(self.first_names), random.choice(self.last_names))
