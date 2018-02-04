import matplotlib.pyplot as plt
import numpy as np

from read_data import read_af_data

from util import balance

data = balance(read_af_data(), y_col='labels')

# Split the data and labels and convert to numpy arrays
# Xs = data['samples'].as_matrix()
# ys = data['labels'].as_matrix()

# Xs = np.array([np.vectorize(int)(x[1:-1].split(',')) for x in Xs])

Xs_0 = data.loc[data['labels'] == 0]['samples']
Xs_1 = data.loc[data['labels'] == 1]['samples']

Xs_0 = np.array([np.vectorize(int)(x[1:-1].split(',')) for x in Xs_0])
Xs_1 = np.array([np.vectorize(int)(x[1:-1].split(',')) for x in Xs_1])

# for x in Xs_0[:5]:
#     plt.plot(x, color='blue')
#
# for x in Xs_1[:5]:
#     plt.plot(x, color='red')

plt.title('Non-AF')
plt.hist(Xs_0[0], bins=30, range=(200, 1700))
plt.show()

plt.title('AF')
plt.hist(Xs_1[0], bins=30, range=(200, 1700))
plt.show()
