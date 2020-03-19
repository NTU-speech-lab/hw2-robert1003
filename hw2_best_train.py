import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from _utils import train_test_split, shuffle, sigmoid, accuracy, cross_entropy_loss, my_dump

# hyperparams
my_seed = 8881003
valid_ratio = 0.01
eps = 1e-8
min_clip_value = 1e-8
max_clip_value = 1 - 1e-8
X_train_fpath = './data/X_train'
y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'

# set random seed
np.random.seed(my_seed)

# read datasets
X_train = pd.read_csv(X_train_fpath, index_col=['id'])
X_test = pd.read_csv(X_test_fpath, index_col=['id'])
y_train = pd.read_csv(y_train_fpath, index_col=['id']).to_numpy().astype('float32').flatten()

# preprocess
X = pd.concat([X_train, X_test])
X = X.drop(columns=[
    ' Not in universe', ' Not in universe.1', ' Not in universe.2', ' Not in universe.3', ' Not in universe.4', ' Not in universe.5',
    ' Not in universe.6', ' Not in universe.7', ' Not in universe.8', ' Not in universe.9', ' Not in universe.10', ' Not in universe.11',
    ' Not in universe.12', ' ?', ' ?.1', ' ?.2', ' ?.3', ' ?.4', ' ?.5', ' ?.6', ' ?.7', ' Do not know', ' Not in universe or children',
    ' Not in labor force', ' Not identifiable', ' Not in universe under 1 year old', ' Foreign born- Not a citizen of U S '
])
X = pd.concat([
    X,
    pd.get_dummies(pd.cut(X['age'], [-np.inf, 20, 40, 50, 70, np.inf], labels=['<20', '20-40', '40-50', '50-70', '70+'])),
    (X['weeks worked in year'] > 0).astype(int).rename('B - weeks worked in year'),
    (X['wage per hour'] > 0).astype(int).rename('B - wage per hour'),
    (X['capital gains'] > 0).astype(int).rename('B - capital gains'),
    (X['dividends from stocks'] > 0).astype(int).rename('B - dividends from stocks')
], axis=1)

train_size = X_train.shape[0]
count = X.nunique()
one = count[count == 1].index
two = count[count == 2].index
three_or_more = count[count >= 3].index
X = X.drop(columns=one)
mean = X[three_or_more].mean()
std = X[three_or_more].std()
X[three_or_more] = (X[three_or_more] - mean) / std

X_train, X_test = X.iloc[:train_size, :], X.iloc[train_size:, :]
X_train = X_train.to_numpy().astype('float32')
X_test = X_test.to_numpy().astype('float32')
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_ratio=valid_ratio)

# save thos params
my_dump(one, 'hw2_best/one')
my_dump(two, 'hw2_best/two')
my_dump(three_or_more, 'hw2_best/three_or_more')
my_dump(mean, 'hw2_best/mean')
my_dump(std, 'hw2_best/std')

# define Logistic Regression Class
class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None
    def f(self, X, w, b):
        return sigmoid(X @ w + b, min_clip_value, max_clip_value)
    def predict(self, X, w, b):
        return np.round(self.f(X, w, b)).astype(np.int)
    def predict_f(self, X):
        return np.round(self.f(X, self.w, self.b)).astype(np.int)
    def gradient(self, X, y, w, b, lamb):
        y_pred = self.f(X, w, b)
        y_diff = y - y_pred
        w_grad = -np.sum(y_diff * X.T, axis=1) - 2 * lamb * w
        b_grad = -np.sum(y_diff)
        return (w_grad, b_grad)
    def fit(self, X_train, y_train, X_valid=None, y_valid=None, epochs=10, batch_size=32, learning_rate=0.1, print_every=1, lamb=0):
        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        w, b = np.zeros((X_train.shape[1],)), np.zeros((1,))
        step, bst, bstf, bt = 1, 0, None, 0
        mw, mb = np.zeros((X_train.shape[1],)), np.zeros((1,))
        vw, vb = np.zeros((X_train.shape[1],)), np.zeros((1,))

        for epoch in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for i in range(0, X_train.shape[0], batch_size):
                X = X_train[i * batch_size:(i + 1) * batch_size]
                y = y_train[i * batch_size:(i + 1) * batch_size]

                w_grad, b_grad = self.gradient(X, y, w, b, lamb)

                w -= learning_rate / np.sqrt(step) * w_grad
                b -= learning_rate / np.sqrt(step) * b_grad

                #step += 1

            y_pred = self.f(X_train, w, b)
            train_loss.append(cross_entropy_loss(y_pred, y_train) / X_train.shape[0])
            train_acc.append(accuracy(np.round(y_pred), y_train))

            if X_valid is not None:
                y_pred = self.f(X_valid, w, b)
                valid_loss.append(cross_entropy_loss(y_pred, y_valid) / X_valid.shape[0])
                valid_acc.append(accuracy(np.round(y_pred), y_valid))

                if valid_acc[-1] > bst:
                    bst = valid_acc[-1]
                    bstf = (w, b)
                    bt = epoch

            if (epoch + 1) % print_every == 0:
                if X_valid is not None:
                    print(epoch + 1, 'train', train_loss[-1], train_acc[-1], 'valid', valid_loss[-1], valid_acc[-1])
                else:
                    print(epoch + 1, 'train', train_loss[-1], train_acc[-1])

        if X_valid is not None:
            self.w = bstf[0]
            self.b = bstf[1]
            print(bt, bst)
        else:
            self.w = w
            self.b = b
        return train_loss, train_acc, valid_loss, valid_acc

model = LogisticRegression()
train_loss, train_acc, valid_loss, valid_acc = model.fit(
    X_train, y_train, X_valid, y_valid, batch_size=512, epochs=10000, print_every=10, learning_rate=0.001, lamb=0
)

np.save('hw2_best/w.npy', model.w)
np.save('hw2_best/b.npy', model.b)
