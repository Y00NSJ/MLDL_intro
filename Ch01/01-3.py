import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from fishes import *


plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = bream_length + smelt_length
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1]*35 + [0]*14 # 1=도미, 0=빙어

"""디폴트 K-최근접 이웃 모델"""
kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
accuracy = kn.score(fish_data, fish_target)
print(accuracy)

# 예측
prediction = kn.predict([[30, 600], [10, 100]])
print(prediction)

"""n_neighbors를 49로 지정"""
kn49 = KNeighborsClassifier(n_neighbors=49)
kn49.fit(fish_data, fish_target)
accuracy49 = kn49.score(fish_data, fish_target)
print(accuracy49)