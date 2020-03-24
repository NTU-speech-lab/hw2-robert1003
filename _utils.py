import numpy as np
from pickle import load, dump

def train_test_split(X, y, test_ratio=0.2, balanced=True):
    if not balanced:
        train_size = int(X.shape[0] * (1 - test_ratio) + 0.5)
        choice = np.random.permutation(X.shape[0])
        print(len(choice))
        return X[choice[:train_size]], X[choice[train_size:]], y[choice[:train_size]], y[choice[train_size:]]
    else:
        b = {}
        for k, z in enumerate(y):
            if z not in b:
                b[z] = []
            b[z].append(k)
        train_index, test_index = [], []
        for k in b:
            b[k] = np.array(b[k])
            np.random.shuffle(b[k])
            train_size = int(len(b[k]) * (1 - test_ratio) + 0.5)
            for _ in range(1 if k else 1):
                train_index.append(b[k][:train_size])
            test_index.append(b[k][train_size:])
        train_index = np.hstack(train_index)
        test_index = np.hstack(test_index)
        return X[train_index], X[test_index], y[train_index], y[test_index]

def shuffle(X, y):
    permu = np.random.permutation(X.shape[0])
    return (X[permu], y[permu])

def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x)) if x >= 0 else np.exp(x) / (1 + np.exp(x))

def sigmoid(z, min_clip_value, max_clip_value):
    #print(list(map(_sigmoid, z)))
    return np.clip(np.array(list(map(_sigmoid, z))), min_clip_value, max_clip_value)
    #return np.clip(1 / (1.0 + np.exp(-z)), min_clip_value, max_clip_value)

def accuracy(y_pred, y_true):
    return 1.0 - np.mean(np.abs(y_pred - y_true))

def cross_entropy_loss(y_pred, y_true):
    return -np.dot(y_true, np.log(y_pred)) - np.dot((1 - y_true), np.log(1 - y_pred))

def my_dump(a, f_name):
    with open(f_name, 'wb') as f:
        dump(a, f)

def my_load(f_name):
    with open(f_name, 'rb') as f:
        a = load(f)
    return a
