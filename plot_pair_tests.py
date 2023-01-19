import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

clfs_b = ['GNB', 'CART', 'MLP']
clfs_a = ['GNB', 'CART']

filenames_b = {'bagging_5': 5, 'bagging_10': 10, 'bagging_15': 15}
filenames_a = {'adaboost_5': 5, 'adaboost_10': 10, 'adaboost_15': 15}

scores_a = [[], [], []]
scores_b = [[], [], []]

for ind, file in enumerate(filenames_a):
    scores_a[ind] = np.load(f'./results_pair_stats/{file}.npy')

for ind, file in enumerate(filenames_b):
    scores_b[ind] = np.load(f'./results_pair_stats/{file}.npy')

fig, ax = plt.subplots(len(clfs_b), len(clfs_b))

score_index = 0
i = 0
eq = ['GNB vs CART', 'CART vs MLP', 'GNB vs MLP']

while score_index < 9:
    score = scores_b[i]

    data = {
        'gnb_cart': [score[0][1], score[1][0]],
        'cart_mlp': [score[1][2], score[2][1]],
        'gnb_mlp': [score[0][2], score[2][0]],
    }

    for x in range(3):
        ax[x][0].set_ylabel(f'{(x + 1) * 5} classifiers')
        ax[x][i].bar(np.arange(2) + x * 0.2, list(data.values())[x], width=0.25)
        ax[x][i].set_title(eq[x])
        ax[x][i].set_xticks([])
        ax[x][i].yaxis.set_major_locator(MaxNLocator(integer=True))
    score_index += 3
    i += 1

fig.suptitle('Bagging: direct comparison in pair tests - summarized')


fig, ax = plt.subplots(1, 3)

for i, score in enumerate(scores_a):
    data = {
        'gnb_cart': [score[0][1], score[1][0]],
    }
    ax[i].set_ylabel(f'{(i + 1) * 5} classifiers')
    ax[i].bar(np.arange(2), list(data.values())[0], width=0.25)
    ax[i].set_title('GNB vs CART')
    ax[i].set_xticks([])
    ax[i].yaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle('AdaBoost: direct comparison in pair tests - summarized')
plt.show()
