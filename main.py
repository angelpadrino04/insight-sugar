

import os
from training import train_and_evaluate

os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)


if __name__ == "__main__":
    train_and_evaluate()
