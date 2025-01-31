from data_adapters.data_adapter import DataAdapter
from datetime import datetime
import random
import calendar


class DateAdapter(DataAdapter):

    FORMATS = [
        '%m/%d/%Y', # 01/01/1970
        '%B %-d, %Y', # January 1, 1970
        '%B %-d' # January 1
    ]

    def sample(self):
        year = random.randint(1300, 2100)
        month = random.randint(1, 12)
        return datetime(
            year,
            month,
            random.randint(1, calendar.monthrange(year, month)[1])
        ).strftime(random.choice(self.FORMATS))
