import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os

from training import train_and_evaluate

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)


if __name__ == "__main__":
    train_and_evaluate()
