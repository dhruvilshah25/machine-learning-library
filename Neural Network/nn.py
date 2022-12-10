# I got some inspiration from https://www.pyimagesearch.com/2021/05/06/backpropagation-from-scratch-with-python/ but the code is my own. 
import numpy as np

class sigmoid:
    def __call__(self, x: float) -> float:
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig

    def deriv(self, x):
        z = np.exp(-x)
        sig = 1 / (1 + z)
        return sig * (1 - sig)

class identity:
    def __call__(self, x: float) -> float:
        return x

    def deriv(self, x):
        return 1

class FCLayer:
    def __init__(self, in_channels, out_channels, activation_function, weight_init, include_bias = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        if activation_function == 'sigmoid': self.activation_function = sigmoid()
        elif activation_function == 'identity': self.activation_function = identity()
        else: raise NotImplementedError
        
        if include_bias:
            shape = (self.in_channels+1, self.out_channels+1)
        else:
            shape = (self.in_channels+1, self.out_channels)

        if weight_init == 'zeroes':
            self.layer_weights = np.zeros(shape, dtype=np.float128)
        elif weight_init == 'random':
            self.layer_weights = np.random.standard_normal(shape)
        else: raise NotImplementedError
            
    def __str__(self) -> str:
        return str(self.layer_weights)
    
    def eval(self, x):
        return self.activation_function(np.dot(x, self.layer_weights))
    
    def backwards(self, zs, partials):
        delta = np.dot(partials[-1], self.layer_weights.T)
        delta *= self.activation_function.deriv(zs)
        return delta
    
    def update_ws(self, lr, zs, partials):
        grad = np.dot(zs.T, partials)
        self.layer_weights += -lr * grad
        return grad

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x): 
        x = np.append(1, x)
        zs = [np.atleast_2d(x)]

        for l in range(len(self.layers)):
            out = self.layers[l].eval(zs[l])
            zs.append(out)

        return float(zs[-1]), zs

    def backward(self, zs, y, lr = 0.1, display = False):

        partials = [zs[-1] - y]

        # we've already done the last layer with the above calculation, so skip it.
        for l in range(len(zs) - 2, 0, -1):
            delta = self.layers[l].backwards(zs[l], partials)
            partials.append(delta)
    
        partials = partials[::-1] # flip the array around

        for l in range(len(self.layers)):
            grad = self.layers[l].update_ws(lr, zs[l], partials[l])
            if display: print(f"gradient of layer {l+1}: \n{grad}")
            