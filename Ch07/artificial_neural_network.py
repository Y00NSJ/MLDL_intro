import keras


# fetch Fashion MNIST dataset
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
print("shapes:", train_input.shape, train_target.shape)
print("shapes:", test_input.shape, test_target.shape)
