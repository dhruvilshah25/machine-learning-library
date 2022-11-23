import pandas as pd
import numpy as np
from scipy.optimize import minimize


def primal(x, y, lr_sched, C=100 / 873, T=100):
    rows, cols = x.shape
    w = np.zeros(cols)
    idx = np.arange(rows)
    for t in range(1, T + 1):
        np.random.shuffle(idx)
        x = x[idx, :]
        y = y[idx]
        lr = lr_sched(t)
        for i in range(rows):
            tmp = y[i] * np.sum(x[i] * w)
            if tmp <= 1:
                g = np.copy(w)
                g[-1] = 0
                w = w - lr * g + C * rows * y[i] * x[i, :]
            else:
                w = (1 - lr) * w
    # transpose w since np.transpose is stupid
    w = np.reshape(w, (-1, 1))
    return w


def dual(x, y, C):
    x_prod = x @ x.T * (y * y[:, np.newaxis])

    def dual_svm_objective(a):
        return .5 * (a.T @ (x_prod @ a)) - np.sum(a)

    def dual_svm_jac(a):
        return (a.T @ x_prod) - np.ones(a.shape[0])

    a0 = np.zeros(x.shape[0])
    constraints = ({'type': 'eq', 'fun': lambda alpha: np.sum(alpha * y),
                    'jac': lambda _: y})
    bounds = np.array(x.shape[0] * [(0, C)])
    res = minimize(dual_svm_objective, a0, constraints=constraints,
                   method='SLSQP', bounds=bounds, jac=dual_svm_jac)

    # Rounding them
    res.x[np.isclose(res.x, 0, atol=.001)] = 0
    res.x[np.isclose(res.x, C, atol=.001)] = C
    w = np.sum(np.reshape(res.x, (-1, 1)) * np.reshape(y, (-1, 1)) * x, axis=0)
    idx = np.where((res.x > 0) & (res.x < C))
    b = np.mean(y[idx] - np.matmul(x[idx, :], np.reshape(w, (-1, 1))))
    w = np.append(w, b)
    return np.reshape(w, (-1, 1))


def gaussian_kernel(x, y, C, gamma):
    x_prod = x * x[:, np.newaxis]
    x_prod = np.exp((-(np.sum(np.square(x_prod), axis=2)) / gamma)) * (y * y[:, np.newaxis])

    def dual_svm_objective(a):
        return .5 * (a.T @ (x_prod @ a)) - np.sum(a)

    def dual_svm_jac(a):
        return (a.T @ x_prod) - np.ones(a.shape[0])

    a_0 = np.zeros(x.shape[0])
    constraints = ({'type': 'eq', 'fun': lambda x_: np.sum(x_ * y),
                    'jac': lambda _: y})
    bounds = np.array(x.shape[0] * [(0, C)])
    res = minimize(dual_svm_objective, a_0, constraints=constraints,
                   method='SLSQP', bounds=bounds, jac=dual_svm_jac)

    res.x[np.isclose(res.x, 0, atol=.001)] = 0
    res.x[np.isclose(res.x, C, atol=.001)] = C
    return res.x


def predict_gk(alpha, train_x, train_y, test_x, gamma):

    predictions = []
    for i in range(test_x.shape[0]):
        sum = 0
        for j in range(train_x.shape[0]):
            sum += alpha[j] * train_y[j] * kernel(train_x[j], test_x[i], gamma)

        predictions.append(np.sign(sum))

    return predictions


def kernel(x_1, x_2, gamma):
    return np.exp((-(np.linalg.norm(x_1 - x_2) ** 2)) / gamma)
