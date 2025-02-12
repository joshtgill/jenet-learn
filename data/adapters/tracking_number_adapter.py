from data.adapters.base_adapter import BaseAdapter
import random
from data.adapters.order_number_adapter import OrderNumberAdapter


class TrackingNumberAdapter(BaseAdapter):

    def sample(self):
        return f'T{OrderNumberAdapter.generate_random_string(self, random.randint(10, 25))}'

