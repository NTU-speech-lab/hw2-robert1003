import numpy as np
import pandas as pd
import sys
from _utils import sigmoid, my_load

# hyperparams
min_clip_value = 1e-8
max_clip_value = 1 - 1e-8

# load pretrained weight
w = np.load('hw2_best/w.npy')
b = np.load('hw2_best/b.npy')

# load test data
X = pd.read_csv(sys.argv[5], index_col=['id'])

# load those params
one = my_load('hw2_best/one')
two = my_load('hw2_best/two')
three_or_more = my_load('hw2_best/three_or_more')
mean = my_load('hw2_best/mean')
std = my_load('hw2_best/std')

# preprocess
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
