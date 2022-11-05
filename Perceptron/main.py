from os import makedirs
import csv
import numpy as np
import perceptron

np.random.seed(33)

try: makedirs("./out/")
except FileExistsError: None



train_x = []
train_y = []
with open("train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(-1 if terms_flt[-1] == 0 else 1)

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open("test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(-1 if terms_flt[-1] == 0 else 1)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("Standard Perceptron")
p = perceptron.Perceptron(train_x, train_y, r=0.1)
print(f"learned weights: {p.weights}")
print(f"testing accuracy: {np.mean(test_y == p.predict(test_x))}")

print("Voted Perceptron")
vp = perceptron.VotedPerceptron(train_x, train_y, r=0.1)
print(f"learned weights and counts: {vp.votes}")
print("making csv of weights")
with open('./out/vp_weights.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['b', 'x1', 'x2', 'x3', 'x4', 'Cm'])
    for w in vp.votes:
        row = w[0]
        row = np.append(row, w[1])
        writer.writerow(row)
print(f"testing accuracy: {np.mean(test_y == vp.predict(test_x))}")

print("Averaged Perceptron")
ap = perceptron.AveragedPerceptron(train_x, train_y, r=0.1)
print(f"learned weights: {ap.weights}")
print(f"testing accuracy: {np.mean(test_y == ap.predict(test_x))}")