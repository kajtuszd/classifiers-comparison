import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


files = [
    'balance-scale',
    'movement_libras',
    'texturaPatch',
    'contraceptive',
    'newthyroid',
    'texture',
    'page-blocks',
    'vehicles',
    'digits',
    'pendigits',
    'satimage',
    'winequality-red',
    'id4-cars',
    'segment',
    'wq',
    'led7digit',
    'shuttle',
    'letter',
    'tae',
    'zoo',
]


clfs_b = ['GNB', 'CART', 'MLP']
clfs_a = ['GNB', 'CART']

filenames_b = {'bagging_5': 5, 'bagging_10': 10, 'bagging_15': 15}
filenames_a = {'adaboost_5': 5, 'adaboost_10': 10, 'adaboost_15': 15}

scores_b = [[], [], []]
scores_a = [[], [], []]

mean_ranks_b = []
mean_ranks_a = []

pair_tests_b = []
pair_tests_a = []

for ind, file in enumerate(filenames_a):
    scores_a[ind] = np.load(f'./results/main/{file}.npy')
    mean_ranks_a.append(np.load(f'./results/mean_ranks/{file}.npy'))
    pair_tests_a.append(np.load(f'./results/pair_stats/{file}.npy'))

for ind, file in enumerate(filenames_b):
    scores_b[ind] = np.load(f'./results/main/{file}.npy')
    mean_ranks_b.append(np.load(f'./results/mean_ranks/{file}.npy'))
    pair_tests_b.append(np.load(f'./results/pair_stats/{file}.npy'))


tbl_b = [[], [], []]
tbl_a = [[], [], []]

tests_better_a = []
tests_better_b = []

for dataset in pair_tests_b:
    for matrix in dataset:
        for row in matrix:
            a = ([str(int(ind + 1)) if int(elem) != 0 else '' for ind, elem in enumerate(row)])
            b = [''.join(a)]
            tests_better_b.append(b if b != [''] else ['-'])

for dataset in pair_tests_a:
    for matrix in dataset:
        for row in matrix:
            a = ([str(int(ind + 1)) if int(elem) != 0 else '' for ind, elem in enumerate(row)])
            b = [''.join(a)]
            tests_better_a.append(b if b != [''] else ['-'])

counter_b = 0

for ind, val in enumerate(scores_b):
    for i, r in enumerate(val):
        tbl_b[ind].append([files[i]] + list(np.round(r, 3)))
        tbl_b[ind].append([''] + [*tests_better_b[counter_b], *tests_better_b[counter_b+1], *tests_better_b[counter_b+2]])
        counter_b += 3
    tbl_b[ind].append(['----------'] * 5 + list(mean_ranks_b[ind]))
    tbl_b[ind].append(['Mean rank'] + list(np.round(mean_ranks_b[ind], 3)))

counter_a = 0

for ind, val in enumerate(scores_a):
    for i, r in enumerate(val):
        tbl_a[ind].append([files[i]] + list(np.round(r, 3)))
        tbl_a[ind].append([''] + [*tests_better_a[counter_a], *tests_better_a[counter_a+1]])
        counter_a += 2
    tbl_a[ind].append(['----------'] * 5 + list(mean_ranks_a[ind]))
    tbl_a[ind].append(['Mean rank'] + list(np.round(mean_ranks_a[ind], 3)))


bagging_dict = dict(zip(filenames_b.keys(), tbl_b))
adaboost_dict = dict(zip(filenames_a.keys(), tbl_a))


for bagging_case, bagging_test in bagging_dict.items():
    print(f"\n{bagging_case}\n" + tabulate(bagging_test, headers=['dataset'] + clfs_b, tablefmt='fancy_grid'))

for adaboost_case, adaboost_test in adaboost_dict.items():
    print(f"\n{adaboost_case}\n" + tabulate(adaboost_test, headers=['dataset'] + clfs_a, tablefmt='fancy_grid'))

