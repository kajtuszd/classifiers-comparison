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

# print("\nScores:\n", scores_a)
# print("\nScores:\n", scores_b)

fig, ax = plt.subplots(1, DATASETS_NUM)

# for estimators_num_ind, estimators_num_example in enumerate(scores_b):
#     print(f'Num of estimators: {(estimators_num_ind + 1) * 5}\n', estimators_num_example)
    # print(estimators_num_example)
    # matrix = np.stack(list(zip(*estimators_num_example)))
    # print(matrix)
    # ax[estimators_num_ind].bar(X + 0.00, matrix[0], color='b', width=0.2)
    # ax[estimators_num_ind].bar(X + 0.2, matrix[1], color='g', width=0.2)
    # ax[estimators_num_ind].bar(X + 0.4, matrix[2], color='r', width=0.2)
    # print(matrix[0])
    # print(matrix[1])
    # print(matrix[2])
#     ax[estimators_num_ind].legend(labels=clfs_b)
#     ax[estimators_num_ind].set_ylabel('Quality (%)')
#     ax[estimators_num_ind].set_xlabel('Datasets')
#     ax[estimators_num_ind].set_title(f'Num of estimators: {(estimators_num_ind + 1)* 5}')
#
# plt.show()



# tabela rozmiaru: liczba datasetow x (3-eksperymenty x 3-klasyfikatory)
formatted_scores = np.zeros((DATASETS_NUM, B_CLF_NUM, EXPERIMENTS_NUM))

# Na poczatku, kolejne warstwy tabeli:
# 1. eksperyment - liczba klasyfikatorow
# 2. uzyta baza danych
# 3. wyniki dla poszczegolnych klasyfikatorow - GNB, CART, MLP

# Na koncu, kolejne warstwy tabeli:
# 1. Baza danych
# 2. macierz 3x3 - eksperyment x klasyfikator

for exp_index, exp_scores in enumerate(scores_b):
    for dataset_index, dataset_scores in enumerate(exp_scores):
        for clf_index, clf_score in enumerate(dataset_scores):
            formatted_scores[dataset_index][clf_index][exp_index] = scores_b[exp_index][dataset_index][clf_index]


# print('---FORMATTED SCORES---')
# print(formatted_scores)
# print('------------------')

for dataset_index, dataset in enumerate(formatted_scores):
    dataset = dataset.T

    ax[dataset_index].bar(np.arange(3), dataset[0], width=0.25)
    ax[dataset_index].bar(np.arange(3) + 0.25, dataset[1], width=0.25)
    ax[dataset_index].bar(np.arange(3) + 0.5, dataset[2], width=0.25)

    ax[dataset_index].legend(labels=['5 clfs', '10 clfs', '15 clfs'])
    ax[dataset_index].set_ylabel('Quality (%)')
    ax[dataset_index].set_xlabel('GNB       CART       MLP')
    ax[dataset_index].set_xticks([])
    ax[dataset_index].set_title(f'dataset no.{dataset_index+1}')

plt.suptitle('Boosting: final results')

# plt.ylim(0.4, 0.9)
# plt.show()


print(scores_a)
formatted_scores = np.zeros((DATASETS_NUM, A_CLF_NUM, EXPERIMENTS_NUM))

fig, ax = plt.subplots(1, DATASETS_NUM)

for exp_index, exp_scores in enumerate(scores_a):
    for dataset_index, dataset_scores in enumerate(exp_scores):
        for clf_index, clf_score in enumerate(dataset_scores):
            formatted_scores[dataset_index][clf_index][exp_index] = scores_a[exp_index][dataset_index][clf_index]



for dataset_index, dataset in enumerate(formatted_scores):
    dataset = dataset.T

    ax[dataset_index].bar(np.arange(2), dataset[0], width=0.25)
    ax[dataset_index].bar(np.arange(2) + 0.25, dataset[1], width=0.25)
    ax[dataset_index].bar(np.arange(2) + 0.5, dataset[2], width=0.25)

    ax[dataset_index].legend(labels=['5 clfs', '10 clfs', '15 clfs'])
    ax[dataset_index].set_ylabel('Quality (%)')
    ax[dataset_index].set_xlabel('GNB       CART')
    ax[dataset_index].set_xticks([])
    ax[dataset_index].set_title(f'dataset no.{dataset_index+1}')

plt.suptitle('AdaBoost: final results')
plt.show()

