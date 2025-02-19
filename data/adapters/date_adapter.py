from data.adapters.base_adapter import BaseAdapter
from datetime import datetime
import random
import calendar


class DateAdapter(BaseAdapter):

    FORMATS = [
        '%m/%d/%Y',   # 01/01/1970
        '%m/%d/%y',   # 01/01/70
        '%-m/%-d/%Y', # 1/1/1970
        '%-m/%-d/%y', # 1/1/70
        '%m-%d-%Y',   # 01-01-1970
        '%m-%d-%y',   # 01-01-70
        '%-m-%-d-%Y', # 1-1-1970
        '%-m-%-d-%y', # 1-1-70
        '%m.%d.%Y',   # 01.01.1970
        '%m.%d.%y',   # 01.01.70
        '%-m.%-d.%Y', # 1.1.1970
        '%-m.%-d.%y', # 1.1.70
        '%B %-d, %Y', # January 1, 1970
        '%B %-d'      # January 1
    ]

    def sample(self, k):
        samples = []
        for _ in range(k):
            year = random.randint(1300, 2100)
            month = random.randint(1, 12)
            samples.append(datetime(
                year,
                month,
                random.randint(1, calendar.monthrange(year, month)[1])
            ).strftime(random.choice(self.FORMATS)))

        return samples
