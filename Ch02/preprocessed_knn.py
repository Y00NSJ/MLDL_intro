from fishes import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

train_input, test_input, train_target, test_target = (
    train_test_split(fish_data, fish_target, stratify=fish_target, random_state=42)) # 디폴트 25%

# kn = KNeighborsClassifier()
# kn.fit(train_input, train_target)

sample = [[25, 150]]
# distances, indexes = kn.kneighbors(sample)

### Standardization
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std
test_scaled = (test_input - mean) / std

new_sample = (sample[0] - mean) / std

kn = KNeighborsClassifier()
kn.fit(train_scaled, train_target)

kn.score(test_scaled, test_target)
print(kn.predict([new_sample]))

distances, indexes = kn.kneighbors([new_sample])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new_sample[0], new_sample[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()