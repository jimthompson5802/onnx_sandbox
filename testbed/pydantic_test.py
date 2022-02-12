import os
import sys

import pydantic
from pydantic import BaseModel, StrictInt
import json
import numpy as np



class LivehveRequest(BaseModel):
    field_1: float
    field_2: StrictInt

    class Config:
        abritrary_types_allowed = True



my_data = {
    'field_2': 1.2,
    'field_1': 99.123
}

my_data_json = json.dumps(my_data)

print(my_data_json)

new_data = LivehveRequest(**json.loads(my_data_json))

print(f'LiveHve request: {new_data}')

