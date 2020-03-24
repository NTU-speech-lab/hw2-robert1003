import numpy as np
import pandas as pd
import sys
from _utils import sigmoid, my_load

# hyperparams
min_clip_value = 1e-8
max_clip_value = 1 - 1e-8

# load pretrained weight
w = np.load('hw2_logistic/w.npy')
b = np.load('hw2_logistic/b.npy')

# load test data
X = pd.read_csv(sys.argv[5], index_col=['id'])

# load those params
one = my_load('hw2_logistic/one')
two = my_load('hw2_logistic/two')
three_or_more = my_load('hw2_logistic/three_or_more')
mean = my_load('hw2_logistic/mean')
std = my_load('hw2_logistic/std')

# preprocess
X = X.drop(columns=one)
X[three_or_more] = (X[three_or_more] - mean) / std
X = X.to_numpy().astype('float32')

# predict
#print(w.shape, X.shape)
y = np.round(
    sigmoid(X @ w + b, min_clip_value, max_clip_value)
).astype(np.int)

# write prediction
df = pd.DataFrame(y).astype(int)
df.columns = ['label']
df['id'] = range(0, X.shape[0])
df = df.reindex(columns=['id', 'label'])
df.to_csv(sys.argv[6], index=None)
