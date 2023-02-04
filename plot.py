import numpy as np
import matplotlib.pyplot as plt

clfs_b = ['GNB', 'CART', 'MLP']
clfs_a = ['GNB', 'CART']

filenames_b = {'bagging_5': 5, 'bagging_10': 10, 'bagging_15': 15}
filenames_a = {'adaboost_5': 5, 'adaboost_10': 10, 'adaboost_15': 15}

scores_a = [[], [], []]
scores_b = [[], [], []]

for ind, file in enumerate(filenames_a):
    scores_a[ind] = np.load(f'./results/main/{file}.npy')

for ind, file in enumerate(filenames_b):
    scores_b[ind] = np.load(f'./results/main/{file}.npy')

B_CLF_NUM = len(clfs_b)
A_CLF_NUM = len(clfs_a)
EXPERIMENTS_NUM = 3
DATASETS_NUM = len(scores_b[0])

fig, ax = plt.subplots(5, 4)


formatted_scores = np.zeros((DATASETS_NUM, B_CLF_NUM, EXPERIMENTS_NUM))


for exp_index, exp_scores in enumerate(scores_b):
    for dataset_index, dataset_scores in enumerate(exp_scores):
        for clf_index, clf_score in enumerate(dataset_scores):
            formatted_scores[dataset_index][clf_index][exp_index] = \
            scores_b[exp_index][dataset_index][clf_index]

i = 0
for dataset_index, dataset in enumerate(formatted_scores):
    dataset = dataset.T
    if i == 4:
        i = 0
    ax[int(dataset_index/4)][i].bar(np.arange(3), dataset[0], width=0.1)
    ax[int(dataset_index/4)][i].bar(np.arange(3) + 0.1, dataset[1], width=0.1)
    ax[int(dataset_index/4)][i].bar(np.arange(3) + 0.2, dataset[2], width=0.1)

    # ax[int(dataset_index/4)][i].legend(labels=['5 clfs', '10 clfs', '15 clfs'])
    if i == 0:
        ax[int(dataset_index/4)][i].set_ylabel('Quality (%)')
    # ax[int(dataset_index/4)][i].set_ylabel('Quality (%)')
    ax[int(dataset_index/4)][i].set_xlabel('GNB       CART       MLP')
    ax[int(dataset_index/4)][i].set_xticks([])
    ax[int(dataset_index/4)][i].set_title(f'dataset no.{dataset_index + 1}', size=10)
    i += 1

plt.suptitle('Boosting: final results')
fig.subplots_adjust(wspace=0.3, hspace=0.3)


formatted_scores = np.zeros((DATASETS_NUM, A_CLF_NUM, EXPERIMENTS_NUM))

fig, ax = plt.subplots(5, 4)

for exp_index, exp_scores in enumerate(scores_a):
    for dataset_index, dataset_scores in enumerate(exp_scores):
        for clf_index, clf_score in enumerate(dataset_scores):
            formatted_scores[dataset_index][clf_index][exp_index] = \
            scores_a[exp_index][dataset_index][clf_index]

i = 0
for dataset_index, dataset in enumerate(formatted_scores):
    dataset = dataset.T
    if i == 4:
        i = 0
    ax[int(dataset_index/4)][i].bar(np.arange(2), dataset[0], width=0.1)
    ax[int(dataset_index/4)][i].bar(np.arange(2) + 0.1, dataset[1], width=0.1)
    ax[int(dataset_index/4)][i].bar(np.arange(2) + 0.2, dataset[2], width=0.1)

    # ax[int(dataset_index/4)][i].legend(labels=['5 clfs', '10 clfs', '15 clfs'])
    if i == 0:
        ax[int(dataset_index/4)][i].set_ylabel('Quality (%)')
    ax[int(dataset_index/4)][i].set_xlabel('GNB           CART')
    ax[int(dataset_index/4)][i].set_xticks([])
    ax[int(dataset_index/4)][i].set_title(f'dataset no.{dataset_index + 1}', size=10)
    i += 1


fig.subplots_adjust(wspace=0.3, hspace=0.3)
plt.suptitle('AdaBoost: final results\n 5 clfs: blue, 10 clfs: orange, 15 clfs: green')
plt.show()

