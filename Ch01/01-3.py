import matplotlib.pyplot as plt
from fish_data import bream_length, bream_weight

plt.scatter(bream_length, bream_weight)
plt.xlabel('length of bream')
plt.ylabel('weight of bream')
plt.show()