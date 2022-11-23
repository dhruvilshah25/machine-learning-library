import svm
import pandas as pd
import numpy as np
from tqdm import tqdm as tqdm
import sys


train_data = pd.read_csv('bank-note/train.csv', header=None)
num_cols = train_data.shape[1]
train_x = np.copy(train_data.values)

train_x[:, num_cols - 1] = 1

train_y = train_data.values[:, num_cols - 1]

train_y = 2 * train_y - 1

test_data = pd.read_csv('bank-note/test.csv', header=None)
num_cols = train_data.shape[1]
test_x = np.copy(test_data.values)
test_x[:, num_cols - 1] = 1
test_y = test_data.values[:, num_cols - 1]
test_y = 2 * test_y - 1

C = np.array([100, 500, 700]) / 873
gamma_0 = 0.1
alpha = 0.1


# Primal
def primal():
    print('Primal')
    print('Learning schedule of gamma_0 / (1 + (gamma_0 / alpha) * t)')
    lr_sched = lambda t: gamma_0 / (1 + gamma_0 / alpha * t)
    for c in C:
        print(f'  C: {c:.5f}')

        w = svm.primal(train_x, train_y, lr_sched, c)
        pred = np.sign(np.matmul(train_x, w))
        num_err = np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
        train_err = num_err / train_y.shape[0] / 2
        pred = np.sign(np.matmul(test_x, w))
        num_err = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1))))
        test_err = num_err / test_y.shape[0] / 2
        print(f'Weights : {np.transpose(w)}')
        print(f'Training Error : {train_err:.4f}')
        print(f'Test Error : {test_err:.4f}')
        print()

    print('Learning schedule of gamma_0 / (1 + t)')
    lr_sched = lambda t: gamma_0 / (1 + t)
    for c in C:
        print(f'  C: {c:.5f}')
        w = svm.primal(train_x, train_y, lr_sched, c)
        pred = np.sign(np.matmul(train_x, w))
        num_err = np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
        train_err = num_err / train_y.shape[0] / 2
        pred = np.sign(np.matmul(test_x, w))
        num_err = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1))))
        test_err = num_err / test_y.shape[0] / 2
        print(f'Weights : {np.transpose(w)}')
        print(f'Training Error : {train_err:.4f}')
        print(f'Test Error : {test_err:.4f}')
        print()


dual_train = train_x[:, [x for x in range(num_cols - 1)]]
dual_test = test_x[:, [x for x in range(num_cols - 1)]]


def dual():
    print('Dual')
    for c in C:
        print(f'  C: {c:.5f}')

        w = svm.dual(dual_train, train_y, c)
        pred = np.sign(train_x @ w)
        num_err = np.sum(np.abs(pred - np.reshape(train_y, (-1, 1))))
        train_err = num_err / train_y.shape[0] / 2
        pred = np.sign(test_x @ w)
        num_err = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1))))
        test_err = num_err / test_y.shape[0] / 2
        print(f'Weights : {np.transpose(w)}')
        print(f'Training Error : {train_err:.4f}')
        print(f'Test Error : {test_err:.4f}')
        print()


def gk():
    print('Gaussian Kernel')
    for c in C:
        print(f' C: {c}')
        old_idx = None
        for gamma in [0.1, 0.5, 1, 5, 100]:
            print(f'Gamma: {gamma}')
            alpha = svm.gaussian_kernel(dual_train, train_y, c, gamma)
            idx = np.where(alpha > 0)[0]
            print('#sv: ', len(idx))
            pred = svm.predict_gk(alpha, dual_train, train_y, dual_train, gamma)
            train_err = 1 - np.mean(pred == train_y)
            pred = svm.predict_gk(alpha, dual_train, train_y, dual_test, gamma)
            test_err = 1 - np.mean(pred == test_y)
            print(f'Training Error: {train_err:.4f}')
            print(f'Test error: {test_err:.4f}')

            if (old_idx is not None):
                intersect = np.intersect1d(old_idx, idx)
                print(f'Intersections: {len(intersect)}')
            old_idx = idx
            print()


which = sys.argv[1] if len(sys.argv) != 1 else False
if(not which or which.lower() == 'primal'):
    primal()

if(not which or which.lower() == 'dual'):
    dual()

if(not which or which.lower() == 'gk'):
    gk()

breakpoint
