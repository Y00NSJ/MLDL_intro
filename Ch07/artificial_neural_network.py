import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

# fetch Fashion MNIST dataset
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# fig, axs = plt.subplots(1, 10, figsize=(10, 10))    # 1개 행, 10개 열
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     axs[i].axis('off')
# plt.show()
# print(train_target[:10])
# print(np.unique(train_target, return_counts=True))

### Classify using Logistic Regression
# SGD Classifier를 사용하기 위해 각 이미지 샘플을 1차원으로 reshape
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)  # 784

# 로지스틱 회귀 및 교차 검증
sc = SGDClassifier(loss='log_loss', max_iter=50, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(f"average test score: {np.mean(scores['test_score'])}")