from fishes import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = (
    train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)) # 디폴트 25%

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

sample = [[25, 150]]
distances, indexes = kn.kneighbors(sample)

plt.scatter(test_input[:, 0], test_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')

plt.xlabel('length')
plt.ylabel('weight')
plt.show()