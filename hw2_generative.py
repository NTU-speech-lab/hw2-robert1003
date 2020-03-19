import numpy as np
import pandas as pd
import sys
from _utils import sigmoid, my_load

# hyperparams
min_clip_value = 1e-8
max_clip_value = 1 - 1e-8

# load pretrained weight
w = np.load('hw2_generative/w.npy')
b = np.load('hw2_generative/b.npy')

# load test data
X = pd.read_csv(sys.argv[5], index_col=['id'])

# predict
print(w.shape, X.shape)
y = np.round(
    sigmoid(X @ w + b, min_clip_value, max_clip_value)
).astype(np.int)

# write prediction
df = pd.DataFrame(1 - y).astype(int)
df.columns = ['label']
df['id'] = range(0, X.shape[0])
df = df.reindex(columns=['id', 'label'])
df.to_csv(sys.argv[6], index=None)
