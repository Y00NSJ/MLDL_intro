from fishes import *
import numpy as np
from sklearn.model_selection import train_test_split


fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = (
    train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)) # 디폴트 25%