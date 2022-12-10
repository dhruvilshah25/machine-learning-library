import nn
import numpy as np
from os import makedirs
import matplotlib.pyplot as plt

try: makedirs("./out/")
except FileExistsError: None

dataset_loc = "data/bank-note/"

def square_loss(pred, target):
    return 0.5*(pred - target)**2

train_x = []
train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : np.float128(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(terms_flt[-1])

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : np.float128(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(terms_flt[-1])

test_x = np.array(test_x)
test_y = np.array(test_y)

def train(num_epochs, net, train_x, train_y, lr_0 = 0.5, d = 1):
    all_losses = []

    for e in range(num_epochs):
        losses = []
        idxs = np.arange(len(train_x))
        np.random.shuffle(idxs)
        for i in idxs:
            y, zs = net.forward(train_x[i])
            losses.append(square_loss(y, train_y[i]))

            lr = lr_0 / (1 + (lr_0/d)*e)
            net.backward(zs, train_y[i], lr)
        print(f"epoch {e+1} training error: {np.mean(losses):>8f}")
        all_losses.append(np.mean(losses))
    
    return all_losses

def test(net, test_x, test_y):
    losses = []
    for i in range(len(test_x)):
        y, _ = net.forward(test_x[i])
        losses.append(square_loss(y, test_y[i]))
    print(f"testing error: {np.mean(losses):>8f}\n")

    return np.mean(losses)

print("5-wide network:\n-------------------------------")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 5, activation_function = 'sigmoid', weight_init='zeroes'), # input
    nn.FCLayer(in_channels = 5, out_channels = 5, activation_function = 'sigmoid', weight_init='zeroes'), # hidden
    nn.FCLayer(in_channels = 5, out_channels = 1, activation_function = 'identity', weight_init='zeroes', include_bias=False) # output
])

training_acc = train(35, net, train_x, train_y, lr_0 = 0.5, d = 1)
fig, ax = plt.subplots()
ax.plot(training_acc)
ax.set_title("width = 5, zeroes init")
ax.set_xlabel("iteration")
ax.set_ylabel("squared error")
plt.savefig("./out/zeroes_w5.png")
testing_acc = test(net, test_x, test_y)

print("10-wide network:\n-------------------------------")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 10, activation_function = 'sigmoid', weight_init='zeroes'), # input
    nn.FCLayer(in_channels = 10, out_channels = 10, activation_function = 'sigmoid', weight_init='zeroes'), # hidden
    nn.FCLayer(in_channels = 10, out_channels = 1, activation_function = 'identity', weight_init='zeroes', include_bias=False) # output
])

training_acc = train(35, net, train_x, train_y, lr_0 = 0.5, d = 1)
fig, ax = plt.subplots()
ax.plot(training_acc)
ax.set_title("width = 10, zeroes init")
ax.set_xlabel("iteration")
ax.set_ylabel("squared error")
plt.savefig("./out/zeroes_w10.png")
testing_acc = test(net, test_x, test_y)

print("25-wide network:\n-------------------------------")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 25, activation_function = 'sigmoid', weight_init='zeroes'), # input
    nn.FCLayer(in_channels = 25, out_channels = 25, activation_function = 'sigmoid', weight_init='zeroes'), # hidden
    nn.FCLayer(in_channels = 25, out_channels = 1, activation_function = 'identity', weight_init='zeroes', include_bias=False) # output
])

training_acc = train(35, net, train_x, train_y, lr_0 = 0.05, d = 1)
fig, ax = plt.subplots()
ax.plot(training_acc)
ax.set_title("width = 25, zeroes init")
ax.set_xlabel("iteration")
ax.set_ylabel("squared error")
plt.savefig("./out/zeroes_w25.png")
testing_acc = test(net, test_x, test_y)

print("50-wide network:\n-------------------------------")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 50, activation_function = 'sigmoid', weight_init='zeroes'), # input
    nn.FCLayer(in_channels = 50, out_channels = 50, activation_function = 'sigmoid', weight_init='zeroes'), # hidden
    nn.FCLayer(in_channels = 50, out_channels = 1, activation_function = 'identity', weight_init='zeroes', include_bias=False) # output
])

training_acc = train(35, net, train_x, train_y, lr_0 = 0.1, d = 1)
fig, ax = plt.subplots()
ax.plot(training_acc)
ax.set_title("width = 50, zeroes init")
ax.set_xlabel("iteration")
ax.set_ylabel("squared error")
plt.savefig("./out/zeroes_w50.png")
testing_acc = test(net, test_x, test_y)

print("100-wide network:\n-------------------------------")
net = nn.NeuralNetwork([
    nn.FCLayer(in_channels = 4, out_channels = 100, activation_function = 'sigmoid', weight_init='zeroes'), # input
    nn.FCLayer(in_channels = 100, out_channels = 100, activation_function = 'sigmoid', weight_init='zeroes'), # hidden
    nn.FCLayer(in_channels = 100, out_channels = 1, activation_function = 'identity', weight_init='zeroes', include_bias=False) # output
])

training_acc = train(35, net, train_x, train_y, lr_0 = 0.01, d = 2)
fig, ax = plt.subplots()
ax.plot(training_acc)
ax.set_title("width = 100, zeroes init")
ax.set_xlabel("iteration")
ax.set_ylabel("squared error")
plt.savefig("./out/zeroes_w100.png")
testing_acc = test(net, test_x, test_y)

plt.show()
