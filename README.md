# NumPy-Neural-Network
A Python neural network implemented mostly from scratch, mainly relying on NumPy

**Motivation**

While building a neural network with libraries like TensorFlow and PyTorch can be very practical and time-efficient, implementing one from scratch can be a more rewarding process and can uncover some of the mystery behind the functions in these libraries. By never designing these functions from scratch, it can be difficult to get the mathematical intuition behind machine learning and understand it from a foundational level. The goal of this project was to build a neural network that classifies handwritten digits without being reliant on libraries like TensorFlow/PyTorch. Calculations, especially those involving tensors, are handled with NumPy. *Disclaimer* - The implementations of these functions use no machine learning libraries like the ones mentioned above, but the loading and preprocessing of the data DOES use functions from TensorFlow and Sklearn. This in no way detracts from the original purpose of the project, which aims to build a machine learning foundation at a low level. 

**The Data**

To help generalize the network design, I wanted to predict handwritten digits on two different datasets. 1: the Sklearn Digits dataset, which has 1797 8 x 8 images of handwritten numbers. 2: the MNIST Digits dataset, which has 60,000 28 x 28 images of handwritten digits (NOTE: Because the MNIST dataset is so large, I took a slice of 6000 of the digits to reduce the computational load)

**Results**

Sklearn Digits (8 x 8) - 98.3% Accuracy

MNIST Digits (28 x 28) - 93.5% Accuracy

**Architecture**

Sklearn: Input → Dense(n) → ReLU → Dense(n) → ReLU → Dense(10) → Softmax

MNIST: Input → Dense(n) → ReLU → Dense(n) → ReLU → Dense(n) → ReLU → Dense(10) → Softmax

**What I implemented**

- Weight Initialization: Weights are initially set as matrices of random numbers according to the desried architecture of the network. He initialization is used to maintain stable activation variance throughout the network, and pairs nicely with the ReLU function.
- ReLU, softmax: Translated these mathematical functions into code used by the network.
- Forward pass: Feeds data through the network using matrix multiplication between layers. The ReLU function is applied to the hidden layers after calculations are performed. The Softmax function is applied to the output to get a clean probability vector used to interpret the predicted digit.
- Cross entropy loss: A function that measures how close the model's digit predictions are to the actual digit
- One-hot-encode: Converts integer labels into one-hot vectors (vectors of length 10 with 9 zeros and a single 1 placed at the index corresponding to the digit in question)
- Backwards pass: Relies on gradient descent and the chain rule from calculus to find where the cross entropy loss is the lowest. Works backwards through the network applying derivatives at each step to tweak the weights in each layer to lower the loss, and get more accurate predictions.
- Training loop: Ties everything together to train the model. Trains data in batches, propogates through the nework, makes predictions, and applies backpropogation to improve the accuracy of the model. \
- Dynamic learning rate: For the MNIST dataset, a changing learning rate can improve accuracy. The learning rate falls by 2% for each epoch so that big nudges happen in the beginning to push the model towards higher accuracy, and smaller nudges happen later on to fine tune the accuracy.
- Matplot visualization: Side by side graphs of loss over epochs and accuracy over epochs to visualize the performance of the models.

**Visualizations for training curves**
Sklearn Digits:
<img width="1200" height="500" alt="Sklearn - 2 hidden layers" src="https://github.com/user-attachments/assets/b944fd51-e61f-41d8-a971-cce4268fbfdb" />

MNIST Digits:
<img width="1200" height="500" alt="Mnist - 3 hidden layers (6000 random, 50 epochs)" src="https://github.com/user-attachments/assets/4922c051-37bb-43cd-939f-4ae5b8246b45" />

**How to run**

1) Navigate to your Python IDE/Coding platform.
2) Copy the Sklearn_Digits.py and the Mnist_Digits.py files into your platform
3) If you see errors like: Traceback (most recent call last):
  File "<main.py>", line 4, in <module>
ModuleNotFoundError: No module named 'sklearn'
In your terminal, navigate to your current python project location and run:
pip install sklearn
Note: replace 'sklean' with the name of the module in the error output: No module named 'sklearn'
5) Run the code.
6) Visulizations will show up after all all epochs are finished.

**Acknowledgments**
https://letsdatascience.com/blog/build-a-neural-network-from-scratch-in-python
Built with reference to the above tutorial, further extended by adjusting to be compatible with MNIST digits, altered network architecture, added an additional hidden layer for MNIST, implemented dynamic learning rate, implemented data visualization.




