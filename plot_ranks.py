import numpy as np
import matplotlib.pyplot as plt


clfs_b = ['GNB', 'CART', 'MLP']
clfs_a = ['GNB', 'CART']

filenames_b = {'bagging_5': 5, 'bagging_10': 10, 'bagging_15': 15}
filenames_a = {'adaboost_5': 5, 'adaboost_10': 10, 'adaboost_15': 15}

scores_a = [[], [], []]
scores_b = [[], [], []]

for ind, file in enumerate(filenames_a):
    scores_a[ind] = np.load(f'./results/mean_ranks/{file}.npy')

for ind, file in enumerate(filenames_b):
    scores_b[ind] = np.load(f'./results/mean_ranks/{file}.npy')

print(scores_a)
print(scores_b)

fig, ax = plt.subplots(1, len(scores_b))

for i, score in enumerate(scores_b):
    ax[i].set_ylabel(f'{(i + 1) * 5} classifiers')
    ax[i].bar(np.arange(3), score, width=0.25)
    ax[i].set_title('GNB  CART  MLP')
    ax[i].set_xticks([])
    # ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))

plt.suptitle('Bagging: average ranks')
# plt.show()

fig, ax = plt.subplots(1, len(scores_a))

for i, score in enumerate(scores_a):
    ax[i].set_ylabel(f'{(i + 1) * 5} classifiers')
    ax[i].bar(np.arange(2), score, width=0.25)
    ax[i].set_title('GNB  CART')
    ax[i].set_xticks([])
    # ax[i].set_xticks(np.arange(2), ['GNB', 'CART'])
    # ax[i].xticks(np.arange(2), ['GNB', 'CART'])

plt.suptitle('AdaBoost: average ranks')
plt.show()



