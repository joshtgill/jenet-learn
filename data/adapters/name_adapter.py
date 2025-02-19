from data.adapters.text_adapter import TextAdapter
import random


class NameAdapter(TextAdapter):

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


    def sample(self, k):
        return [
            random.choice(self.FORMATS).format(
                random.choice(self.srcs[0]),
                random.choice(self.srcs[1])
            )
            for _ in range(k)
        ]
