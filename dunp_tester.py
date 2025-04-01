import pandas as pd 
import numpy as np
import json
import os


with open("data/mp.2019.04.01.json", "r") as f:
    data = json.load(f)


print(data[1])