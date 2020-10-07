"""Neural network model."""

from typing import Sequence

import numpy as np
class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a cross-entropy loss function and
    L2 regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are passed through
    a softmax, and become the scores for each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix (D, H)
            X: the input data (N, D)
            b: the bias

        Returns:
            the output (N, H)
        """
        # TODO: implement me
        
        return X.dot(W) + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        # TODO: implement me
        # return np.where(X < 0, 0 , X)
        return np.maximum(X,0)

    def softmax(self, X: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data (N , C)

        Returns:
            the output
        """
        # TODO: implement me        
        exp_X = np.exp(X - np.max(X, axis = 1).reshape(-1,1))
        return exp_X / np.sum(exp_X, axis = 1).reshape(-1,1)


    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.softmax in here.

        self.outputs = {}
        output = X
        for i in range(1, self.num_layers + 1 ):
            # linear layer
            output = self.linear(self.params["W" + str(i)], output, self.params["b"+ str(i)])
            self.outputs["linear" + str(i)] = output
            # relu layer
            if (i < self.num_layers):    
                output = self.relu(output)
                self.outputs["relu" + str(i)] = output
            # softmax 
            else:
                output = self.softmax(output)
                self.outputs["softmax"] = output


        return output

    def relu_grad(self, X: np.ndarray, G: np.ndarray, i) -> np.ndarray:
        """
        Calculate the gradient matrix of relu 
        
        Parameters:
            X: Input data of shape(N, H_i).
            G: Upstream gradient
            i: ith layer

        Returns:
            gradient of X
        """
        return np.where(X < 0, 0, G) 

    def linear_grad(self, X, G, i):
        """
        Calculate the gradient matrix of linear layer 
        
        Parameters:
            X: Input data of shape(N, H_i).
            G: Upstream gradient
            i: ith layer 

        Returns:
            W_grad: gradient of W 
            b_grad: gradient of b
            G: gradient of X
        """
        W_grad = X.T.dot(G)
        b_grad = np.sum(G, axis = 0)
        G =  G.dot(self.params["W" + str(i)].T)
        return W_grad, b_grad, G

    def softmax_grad(self, X,Y, reg):
        """
        Calculate the gradient matrix of softmax function
        
        Parameters:
            X: Input data of shape(N, C).
            Y: Input labels of shape(1,C)

        Returns:
            gradient of X
        """
        N = np.shape(X)[0]
        X_tmp = X.copy()
        X_tmp[np.arange(N), Y]-= 1
        X_tmp /= N
        return X_tmp

    def update_grad(self, lr):
        for i in range (1 ,self.num_layers+1):
            W = "W" + str(i)
            b = "b" + str(i)    
            self.params[W] -= self.gradients[W] * lr
            self.params[b] -= self.gradients[b] * lr
            
    def get_loss(self, output, Y, reg):
        """
        Calculate loss

        Parameters:
            X: the input data (N , C)

        Returns:
            single number
        """
        loss = 0.0
        # cross_entropy loss
        N = np.shape(output)[0]
        loss = np.sum(-np.log(output[np.arange(N), Y])) / N

        for i in range(1, self.num_layers + 1):
            W = self.params["W" + str(i)]
            loss += np.sum(W*W) * reg *0.5 / N

        return loss 

    def get_accuracy(self, output, y):
        return np.average(np.argmax(output, axis = 1) == y)


    def backward(
        self, X: np.ndarray, y: np.ndarray, lr: float, reg: float = 0.0
    ) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each layer
        # in self.gradients if you want to be able to debug your gradients
        # later. You can use the same keys as self.params. You can add
        # functions like self.linear_grad, self.relu_grad, and
        # self.softmax_grad if it helps organize your code.

        output = self.outputs["softmax"]
        loss = self.get_loss(output, y, reg)
        self.outputs["relu" + str(0)] = X

        for i in range(self.num_layers, 0, -1):
            W = "W" + str(i)
            b = "b" + str(i)
            # backward softmax layer
            if (i == self.num_layers):
                G = self.softmax_grad( output, y,reg)

            # backward relu layer
            else:
                X_input = self.outputs["linear" + str(i)]
                G = self.relu_grad(X_input, G, i)

            # backward linear layer
            X_input = self.outputs["relu" +str(i-1)] 
            W_grad, b_grad, G = self.linear_grad(X_input, G, i)
            
            # save gradients
            self.gradients[W] = W_grad + reg * self.params[W] 
            self.gradients[b] = b_grad 

        # SGD
        self.update_grad(lr)
        
        return loss
