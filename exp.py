import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import ttest_rel, rankdata
from tabulate import tabulate
from datetime import datetime

start_time = datetime.now()

files = [
    'balance-scale.csv',
    'movement_libras.csv',
    'texturaPatch.csv',
    'contraceptive.csv',
    'newthyroid.csv',
    'texture.csv',
    'page-blocks.csv',
    'vehicles.csv',
    'digits.csv',
    'pendigits.csv',
    'satimage.csv',
    'winequality-red.csv',
    'id18-o-ring-erosion-or-blowby.csv',
    'vowel.csv'
    'id4-cars.csv',
    'segment.csv',
    'wq.csv',
    'led7digit.csv',
    'shuttle.csv',
    'letter.csv',
    'tae.csv',
    'zoo.csv',
]


def ensemble_tests(files, num_of_estimators, EnsembleClass, classifiers, result_filename):
    alpha = .05
    n_folds = 5
    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))
    means = np.zeros((len(files), len(classifiers)))
    ranks = []
    pair_tests = np.zeros((len(files), len(classifiers), len(classifiers)))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1234)
    for file_ind, file in enumerate(files):
        dataset = np.genfromtxt(f"datasets_multi/{file}", delimiter=",")
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        scores = np.zeros((len(classifiers), n_folds))

        # divide dataset to folds
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ensemble learning
            for clf_index, (clf_name, clf) in enumerate(classifiers.items()):
                e_clf = EnsembleClass(estimator=clone(clf),
                                         n_estimators=num_of_estimators,
                                         random_state=0).fit(X_train, y_train)
                predict = e_clf.predict(X_test)
                scores[clf_index, fold] = accuracy_score(y_test, predict)

        # mean classifier scores
        print(f"\nDataset file: {file}")
        print(scores)

        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)
        means[file_ind] = mean

        for clf_id, (clf_name, clf) in enumerate(classifiers.items()):
            print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

        # pair statistical tests
        for i in range(len(classifiers)):
            for j in range(len(classifiers)):
                t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i],
                                                             scores[j])
        headers = [key for key in classifiers.keys()]
        names_column = np.array([[key] for key in classifiers.keys()])
        advantage = np.zeros((len(classifiers), len(classifiers)))
        advantage[t_statistic > 0] = 1
        significance = np.zeros((len(classifiers), len(classifiers)))
        significance[p_value <= alpha] = 1
        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        pair_tests[file_ind] = stat_better
        print("Statistically significantly better:\n", stat_better_table)
    for m in means:
        ranks.append(rankdata(m).tolist())
    ranks = np.array(ranks)
    mean_ranks = np.mean(ranks, axis=0)
    print(mean_ranks)
    np.save(f'./results/mean_ranks/{result_filename}', mean_ranks)
    np.save(f'./results/main/{result_filename}', means)
    np.save(f'./results/pair_stats/{result_filename}', pair_tests)


filenames_b = {
    'bagging_5': 5,
    'bagging_10': 10,
    'bagging_15': 15
}
filenames_a = {
    'adaboost_5': 5,
    'adaboost_10': 10,
    'adaboost_15': 15
}

bagging_clfs = {
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=1234),
    'MLP': MLPClassifier(),
}
print(f"**********\nBAGGING\n**********")

for file, num in filenames_b.items():
    ensemble_tests(files=files, num_of_estimators=num, EnsembleClass=BaggingClassifier, classifiers=bagging_clfs, result_filename=file)


adaboost_clfs = {
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(random_state=1234),
}
print(f"**********\nADABOOST\n**********")

for file, num in filenames_a.items():
    ensemble_tests(files=files, num_of_estimators=num, EnsembleClass=AdaBoostClassifier, classifiers=adaboost_clfs, result_filename=file)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
