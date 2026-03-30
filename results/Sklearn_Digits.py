import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

def initialize_weights(layers):
    #He initialization for ReLU
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

    #output layer: Linear -> Softmax
    cache['Z3'] = cache['A2'] @ params['W3'] + params['b3']
    cache['A3'] = softmax(cache['Z3'])

    return cache['A3'], cache

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
    #Backprop
    N = y_true.shape[0]
    grads = {}

    # Output layer gradient (softmax + cross-entropy)
    dZ3 = (y_pred - y_true) / N
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


def train(X_train, y_train, X_val, y_val, architecture, epochs=200, lr=0.1, batch_size=64):
    #Train with mini-batch gradient descent
    params = initialize_weights(architecture)
    y_train_oh = one_hot_encode(y_train, 10)
    N = X_train.shape[0]
    history = {'train_loss': [],'val_acc':[]}

    for epoch in range(epochs):
        #shuffle training data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X_train[indices]
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

            #update weights
            for key in params:
                params[key] -= lr * grads[f'd{key}']

        #track metrics
        avg_loss = epoch_loss / num_batches
        val_pred, _ = forward(X_val, params)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
            print("||dW1||", np.linalg.norm(grads['dW1']))
            print("||dW2||", np.linalg.norm(grads['dW2']))
            print("||dW3||", np.linalg.norm(grads['dW3']))

    return params, history

#Load and preprocess
digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y))}")

# Train the network
architecture = [64, 256, 64, 10]
params, history = train(X_train, y_train, X_val, y_val,
                        architecture, epochs=200, lr=0.1, batch_size=64)

print(f"\nFinal validation accuracy: {history['val_acc'][-1]:.4f}")

#Plot performance data
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