import nn
import numpy as np

print("testing net")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 5, activation_function = 'sigmoid', weight_init='random'), # input
    nn.FCLayer(in_channels = 5, out_channels = 5, activation_function = 'sigmoid', weight_init='random'), # hidden
    nn.FCLayer(in_channels = 5, out_channels = 1, activation_function = 'identity', weight_init='random', include_bias=False) # output
])

x = np.array([-1.7582,2.7397,-2.5323,-2.234])
ystar = 1
y, A = net.forward(x)
net.backward(A, ystar, display=True)
