import keras
import matplotlib.pyplot as plt


# fetch Fashion MNIST dataset
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
fig, axs = plt.subplots(1, 10, figsize=(10, 10))    # 1개 행, 10개 열
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()
print(train_target[:10])