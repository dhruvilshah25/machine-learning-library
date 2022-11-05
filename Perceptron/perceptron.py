import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = np.ndarray
        
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.weights = np.ndarray
        self.train(X, y, r, epochs)

    def append_bias(self, X):
        return np.insert(X, 0, [1]*len(X), axis=1)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(self.weights, X[i]) <= 0:
                    self.weights += r*(y[i]*X[i])
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        pred = lambda d : np.sign(np.dot(self.weights, d))
        return np.array([pred(xi) for xi in X])

class VotedPerceptron(Perceptron):
    def __init__(self, X, y, r:float = 1e-3, epochs: int=10):
        self.votes = np.ndarray
        self.train(X, y, r, epochs)

    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        m = 0
        weights = [np.zeros_like(X[0])]
        cs = [0]

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(weights[m], X[i]) <= 0:
                    weights[m] += r*(y[i]*X[i])
                    weights.append(weights[m].copy())
                    m += 1
                    cs.append(1)
                else: cs[m] += 1

        self.votes = np.array(list(zip(weights, cs)), dtype=object)
    
    def predict(self, X) -> np.ndarray:
        X = self.append_bias(X)
        preds = np.zeros(len(X), dtype=int)
        for i in range(len(preds)):
            inner = 0
            for w, c in self.votes:
                inner += c * np.sign(np.dot(w, X[i]))
            preds[i] = np.sign(inner)
        return preds

class AveragedPerceptron(Perceptron):
    def train(self, X, y, r:float=1e-3, epochs: int=10):
        X = self.append_bias(X)
        self.weights = np.zeros_like(X[0])
        weights = np.zeros_like(X[0])

        for e in range(epochs):
            idxs = np.arange(len(X))
            np.random.shuffle(idxs)
            for i in idxs:
                if y[i] * np.dot(weights, X[i]) <= 0:
                    weights += r*(y[i]*X[i])
                self.weights = self.weights + weights