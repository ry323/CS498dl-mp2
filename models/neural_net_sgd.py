from typing import Sequence

import numpy as np


class NeuralNetwork_adam:

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):

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

        self.outputs = {}
        self.gradients = {}
        self.m = None
        self.v = None
        self.iter = 0



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
        return np.where(X < 0, 0 , X)

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
        self.outputs[0] = X
        cur = X
        for i in range(1, self.num_layers):
            self.outputs[i] = input
            w = self.params["W" + str(i)]
            b = self.params["b"+ str(i)]
            self.outputs[i] = self.relu(self.linear(w, cur, b))
            cur = self.outputs[i]
        w = self.params["W" + str(self.num_layers)]
        b = self.params["b"+ str(self.num_layers)]
        self.outputs[self.num_layers] = self.linear(w, cur, b)

        return self.softmax(self.outputs[self.num_layers])

    
    def get_accuracy(self, output, y):
        return np.average(np.argmax(output, axis = 1) == y)
    
    def calc_loss(self, N, score_right):
        loss = -np.sum(np.log(score_right))
        loss /= N
        for i in range(1, self.num_layers + 1):
            loss += 1 / N * 1 / 2 * self.reg * np.sum(self.params["W" + str(i)]**2)
        return loss
    
    def softmax_loss_gradient(self, N, y, score):
        sc = score.copy()
        sc[np.arange(N), y] -= 1
        sc /= N
        return sc
    
    def update_gradients(self, N, y, score):
        sl_gradient = self.softmax_loss_gradient(N, y, score)

        final_output = self.outputs[self.num_layers-1]
        grad_w = np.dot(final_output.T, sl_gradient)
        grad_b = np.sum(sl_gradient, axis = 0)   
        self.gradients["W" + str(self.num_layers)] = grad_w
        self.gradients["b" + str(self.num_layers)] = grad_b
        
        lh_gradient = sl_gradient
        for i in range(self.num_layers - 1, 0, -1):
            output_i = self.outputs[i]
            output_i = (output_i > 0).astype(int)
            lr_gradient = (np.dot(lh_gradient, (self.params["W" + str(i+1)]).T)) * output_i
            
            grad_w = np.dot((self.outputs[i-1]).T, lr_gradient) + self.reg * self.params["W" + str(i)]
            grad_b = np.sum(lr_gradient, axis = 0)
            self.gradients["W" + str(i)] = grad_w
            self.gradients["b" + str(i)] = grad_b
            lh_gradient = lr_gradient

    def backward(self, X, y, lr, reg):
        loss = 0.0
        N, D = X.shape
        self.lr = lr
        self.reg = reg
        
        score = self.softmax(self.outputs[self.num_layers]) 
        loss = self.calc_loss(N, score[np.arange(N), y])
        self.update_gradients(N, y, score)
    
        for i in range (1 ,self.num_layers+1):
            W = "W" + str(i)
            b = "b" + str(i)    
            self.params[W] -= self.gradients[W] * lr
            self.params[b] -= self.gradients[b] * lr

        return loss