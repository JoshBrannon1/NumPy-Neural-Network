import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os

def initialize_weights(layers):
    #He intitialization for ReLU
    params = {}
    for i in range(1, len(layers)):
        params[f'W{i}'] = np.random.randn(layers[i-1], layers[i]) * np.sqrt(2/layers[i-1])
        params[f'b{i}'] = np.zeros((1, layers[i]))
    return params

def relu(z):
    return np.maximum(0,z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

def forward(X, params):
    cache = {'A0' : X}

    #Hidden layer 1: Linear -> ReLU
    cache['Z1'] = X @ params['W1'] + params['b1']
    cache['A1'] = relu(cache['Z1'])

    #Hidden layer 2: Linear -> ReLU
    cache['Z2'] = cache['A1'] @ params['W2'] + params['b2']
    cache['A2'] = relu(cache['Z2'])

    #Hidden layer 3: Linear -> ReLU
    cache['Z3'] = cache['A2'] @ params['W3'] + params['b3']
    cache['A3'] = relu(cache['Z3'])

    #output layer: Linear -> Softmax
    cache['Z4'] = cache['A3'] @ params['W4'] + params['b4']
    cache['A4'] = softmax(cache['Z4'])

    return cache['A4'], cache

def cross_entropy_loss(y_pred, y_true):
    #compute cross entropy loss
    N = y_true.shape[0]
    #clip predictions to avoid log(0)
    y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)
    loss = -np.sum(y_true * np.log(y_pred_clipped))/N
    return loss

def one_hot_encode(y, num_classes):
    #convert integer labels to one-hot vectors
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def backward(y_pred, y_true, cache, params):
    #backprop
    N = y_true.shape[0]
    grads = {}

    # Output layer gradient (softmax + cross-entropy shortcut)
    dZ4 = (y_pred - y_true) / N
    grads['dW4'] = cache['A3'].T @ dZ4
    grads['db4'] = np.sum(dZ4, axis=0, keepdims=True)

    # Hidden layer 3
    dA3 = dZ4 @ params['W4'].T
    dZ3 = dA3 * (cache['Z3'] > 0)  # ReLU derivative
    grads['dW3'] = cache['A2'].T @ dZ3
    grads['db3'] = np.sum(dZ3, axis=0, keepdims=True)

    # Hidden layer 2
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * (cache['Z2'] > 0)  # ReLU derivative
    grads['dW2'] = cache['A1'].T @ dZ2
    grads['db2'] = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer 1
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * (cache['Z1'] > 0)  # ReLU derivative
    grads['dW1'] = cache['A0'].T @ dZ1
    grads['db1'] = np.sum(dZ1, axis=0, keepdims=True)

    return grads

def train(x_train, y_train, X_val, y_val, architecture, epochs=200, lr=0.1, batch_size=784):
    #Train with mini-batch gradient descent
    params = initialize_weights(architecture)
    y_train_oh = one_hot_encode(y_train, 10)
    N = x_train.shape[0]
    history = {'train_loss': [],'val_acc':[]}
    initial_lr = lr

    for epoch in range(epochs):
        #shuffle training data each epoch
        indices = np.random.permutation(N)
        X_shuffled = x_train[indices]
        y_shuffled = y_train_oh[indices]

        epoch_loss = 0
        num_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size,N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            #forward pass
            y_pred, cache = forward(X_batch, params)

            #compute loss
            loss = cross_entropy_loss(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1

            #backward pass
            grads = backward(y_pred, y_batch, cache, params)

            #update learning rate
            lr = initial_lr * (0.98 ** epoch)
            #update weights
            for key in params:
                params[key] -= lr * grads[f'd{key}']

        #track metrics
        avg_loss = epoch_loss / num_batches
        val_pred, _ = forward(X_val, params)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            print("||dW1||", np.linalg.norm(grads['dW1']))
            print("||dW2||", np.linalg.norm(grads['dW2']))
            print("||dW3||", np.linalg.norm(grads['dW3']))
            print("||dW4||", np.linalg.norm(grads['dW4']))

    return params, history


#Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixels from 0-255 to 0-1
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Flatten 28x28 -> 784
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# One-hot encode labels
num_classes = 10
y_train_oh = one_hot_encode(y_train, num_classes)
y_test_oh = one_hot_encode(y_test, num_classes)

# Update architecture
architecture = [784, 512, 256, 128, 10]
params = initialize_weights(architecture)

x_val = x_test
y_val = y_test

#Take a random slice of the data to decrease computation needed
indices = np.random.permutation(len(x_train))
x_train = x_train[indices]
y_train = y_train[indices]

x_train = x_train[:10000]
y_train = y_train[:10000]
x_val = x_val[:2000]
y_val = y_val[:2000]

# Train the network
params, history = train(x_train, y_train, x_val, y_val,
                        architecture, epochs=40, lr=0.1, batch_size=64)

print(f"Training samples: {x_train.shape[0]}")
print(f"Validation samples: {x_val.shape[0]}")
print(f"Features: {x_train.shape[1]}, Classes: {len(np.unique(y_val))}")

print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")

#Plot the performance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

os.makedirs("results", exist_ok=True)
# Loss curve
ax1.plot(history['train_loss'])
ax1.set_title("Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Validation accuracy curve
ax2.plot(history['val_acc'], color='orange')
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.show()