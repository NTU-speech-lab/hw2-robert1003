import numpy as np
import pandas as pd
from _utils import train_test_split, shuffle, sigmoid, accuracy, cross_entropy_loss, my_dump

# hyperparams
my_seed = 881003
valid_ratio = 0.3
eps = 1e-8
min_clip_value = 1e-20
max_clip_value = 1 - 1e-20
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
X_train = X_train.to_numpy().astype('float64')
X_test = X_test.to_numpy().astype('float64')
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_ratio=valid_ratio)

# Check shapes
print(f'X_train: {X_train.shape}, X_valid: {X_valid.shape}, y_train: {y_train.shape}, y_valid: {y_valid.shape}, X_test: {X_test.shape}')

# define Generative Model Class
class GenerativeModel:
    def __init__(self):
        self.w = None
        self.b = None
    def f(self, X, w, b):
        return sigmoid(X @ w + b, min_clip_value, max_clip_value)
    def predict(self, X, w, b):
        return np.round(self.f(X, w, b)).astype(np.int)
    def predict_f(self, X):
        return np.round(self.f(X, self.w, self.b)).astype(np.int)
    def fit(self, X_train, y_train, X_valid=None, y_Valid=None):
        # Compute in-class mean
        X_train_0 = np.array([x for x, y in zip(X_train, y_train) if y == 0])
        X_train_1 = np.array([x for x, y in zip(X_train, y_train) if y == 1])

        mean_0 = np.mean(X_train_0, axis = 0)
        mean_1 = np.mean(X_train_1, axis = 0)

        # Compute in-class covariance
        cov_0 = np.zeros((X_train.shape[1], X_train.shape[1]))
        cov_1 = np.zeros((X_train.shape[1], X_train.shape[1]))

        for x in X_train_0:
            cov_0 += np.dot(np.transpose([x - mean_0]), [x - mean_0]) / X_train_0.shape[0]
        for x in X_train_1:
            cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / X_train_1.shape[0]

        # Shared covariance is taken as a weighted average of individual in-class covariance.
        cov = (cov_0 * X_train_0.shape[0] + cov_1 * X_train_1.shape[0]) / (X_train_0.shape[0] + X_train_1.shape[0])

        # Compute inverse of covariance matrix.
        # Since covariance matrix may be nearly singular, np.linalg.inv() may give a large numerical error.
        # Via SVD decomposition, one can get matrix inverse efficiently and accurately.
        u, s, v = np.linalg.svd(cov, full_matrices=False)
        inv = np.matmul(v.T * 1 / s, u.T)

        # Directly compute weights and bias
        w = np.dot(inv, mean_0 - mean_1)
        b =  (-0.5) * np.dot(mean_0, np.dot(inv, mean_0)) + 0.5 * np.dot(mean_1, np.dot(inv, mean_1))\
            + np.log(float(X_train_0.shape[0]) / X_train_1.shape[0])

        # Compute accuracy on training set
        y_train_pred = 1 - self.predict(X_train, w, b)
        print('Training accuracy: {}'.format(accuracy(y_train_pred, y_train)))
        if X_valid is not None:
            y_valid_pred = 1 - self.predict(X_valid, w, b)
            print('Validate accuracy: {}'.format(accuracy(y_valid_pred, y_valid)))
        self.w = w
        self.b = b

model = GenerativeModel()
model.fit(X_train, y_train, X_valid, y_valid)

np.save('hw2_generative/w.npy', model.w)
np.save('hw2_generative/b.npy', model.b)
