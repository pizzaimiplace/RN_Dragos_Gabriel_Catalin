import numpy as np
from torchvision.datasets import MNIST


def download_mnist(is_train: bool):
    dataset = MNIST(root='./ data',
                    transform = lambda x: np.array(x).flatten(),
                    download = True,
                    train = is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)

train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
print(f"Test data shape: {test_X.shape}, Test labels shape: {test_Y.shape}")

def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


def softmax(z):
    result = np.exp(z - np.max(z, axis=1, keepdims=True))
    return result / np.sum(result, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))


def update_weights(X, Y_true, Y_pred, W, b, learning_rate=0.01):
    m = X.shape[0]
    dW = np.dot(X.T, (Y_true - Y_pred)) / m
    db = np.mean(Y_pred - Y_true, axis=0)

    W += learning_rate * dW
    b += learning_rate * db
    return W, b


np.random.seed(42)
W = np.random.randn(784, 10) * 0.01
b = np.zeros(10)


def train_perceptron(train_X, train_Y, test_X, test_Y, epochs=200, batch_size=100, learning_rate=0.01):
    global W, b
    train_size = train_X.shape[0]

    for epoch in range(epochs):
        indices = np.random.permutation(train_size)
        train_X, train_Y = train_X[indices], train_Y[indices]

        for start in range(0, train_size, batch_size):
            end = start + batch_size
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]

            z = np.dot(X_batch, W) + b
            Y_pred = softmax(z)

            W, b = update_weights(X_batch, Y_batch, Y_pred, W, b, learning_rate)

        z_train = np.dot(train_X, W) + b
        Y_train_pred = softmax(z_train)
        loss_train = cross_entropy_loss(Y_train_pred, train_Y)

        z_test = np.dot(test_X, W) + b
        Y_test_pred = softmax(z_test)
        test_accuracy = np.mean(np.argmax(Y_test_pred, axis=1) == np.argmax(test_Y, axis=1))

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss_train:.4f} - Test Accuracy: {test_accuracy:.4f}")


train_Y_one_hot = one_hot_encode(train_Y)
test_Y_one_hot = one_hot_encode(test_Y)
print("Shape of train_Y_one_hot:", train_Y_one_hot.shape)
print("Shape of test_Y_one_hot:", test_Y_one_hot.shape)
train_perceptron(train_X, train_Y_one_hot, test_X, test_Y_one_hot, epochs=200, batch_size=100, learning_rate=0.01)



