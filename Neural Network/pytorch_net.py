import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np

class RegressionDataset(Dataset):
    def __init__(self, datafile):
        xs = []
        ys = []
        with open(datafile, "r") as f:
            for line in f:
                terms = line.strip().split(",")
                terms_flt = list(map(lambda x : np.float32(x), terms))
                xs.append(terms_flt[:-1])
                ys.append(terms_flt[-1])

        self.xs = np.array(xs)
        self.ys = np.array(ys)
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

train_data = RegressionDataset('data/bank-note/train.csv')
test_data = RegressionDataset('data/bank-note/train.csv')

# Create data loaders.
batch_size = 10
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for x, y in test_dataloader:
    print("x:", x)
    print("y:", y)
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(torch.reshape(pred, y.shape), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            train_loss.append(loss.item())

    print(f"training error: {np.mean(train_loss):>8f}")
    return train_loss

def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(torch.reshape(pred, y.shape), y).item()
    test_loss /= num_batches
    print(f"test error: {test_loss:>8f} \n")

def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)

def init_he(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)

widths = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
activations = [(nn.ReLU(), init_he, "ReLU"), (nn.Tanh(), init_xavier, "Tanh")]

for ac_fn, init_fn, ac_name in activations:
    print(f"using activation function {ac_name}")
    for width in widths:
        for depth in depths:

            print(f"{depth}-deep, {width}-wide network:\n-------------------------------")
            # Define model
            class NeuralNetwork(nn.Module):
                def __init__(self):
                    super(NeuralNetwork, self).__init__()
                    self.input = nn.Sequential(nn.Linear(4, width), ac_fn)
                    self.body = nn.ModuleList([])
                    for i in range(depth-2):
                        self.body.append(nn.Sequential(nn.Linear(width, width), ac_fn))
                    self.out = nn.Linear(width, 1)

                def forward(self, x):
                    x = self.input(x)
                    for layer in self.body:
                        x = layer(x)
                    res = self.out(x)
                    return res

            model = NeuralNetwork().to(device)
            model.apply(init_fn)

            loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            train_losses = np.array([])
            epochs = 15
            for t in range(epochs):
                print(f"epoch {t+1}", end=' ')
                epoch_losses = train(train_dataloader, model, loss_fn, optimizer)
                train_losses = np.append(train_losses, epoch_losses)

            fig, ax = plt.subplots()
            ax.plot(train_losses)
            ax.set_title(f"PyTorch: {depth}-deep, {width}-wide network")
            ax.set_xlabel("iteration")
            ax.set_ylabel("squared error")
            plt.savefig(f"./out/torch_{ac_name}_d{depth}_w{width}.png")
            plt.close()
            
            test(test_dataloader, model, loss_fn)

print("Done!\nPlots saved in './out/'")