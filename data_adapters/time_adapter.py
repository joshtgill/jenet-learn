from data_adapters.data_adapter import DataAdapter
from datetime import time
import random


class TimeAdapter(DataAdapter):

    FORMATS = [
        '%H:%M', # 13:00
        '%-I:%M', # 1:00
        '%-I:%M %p', # 1:00 PM
    ]

    def sample(self):
        return time(
            hour=random.randint(0, 23),
            minute=random.randint(0, 59)
        ).strftime(random.choice(self.FORMATS))
