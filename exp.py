import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import ttest_rel
from tabulate import tabulate


files = [
    # 'aids.csv',
    # 'amazon_employee_access.csv',
    # 'balance-scale.csv',
    # 'chorobySercaAfrica.csv',
    # 'diabetes.csv',
    # 'digits.csv',
    # 'ecoli.csv',
    # 'fruitfly.csv',
    # 'id12-Statlog_australia.csv',
    # 'id14-brest-cancer.csv',
    # 'id18-o-ring-erosion-or-blowby.csv',
    # 'id4-cars.csv',
    # 'id6-Ionosphere.csv',
    # 'id8-congress-vote-record.csv',
    # 'monks.csv',
    # 'pendigits.csv',
    # 'phishing.csv',
    # 'plamyOleju.csv',
    # 'shuttle.csv'
    # 'stock.csv',
    # 'texturaPatch.csv',
    # 'ttt.csv',
    # 'vehicles.csv',


    ## NOWE PONIZEJ Z KEELA
    # 'vowel.csv',
    # 'texture.csv',
    # 'movement_libras.csv',
    # 'letter.csv',
    # 'led7digit.csv',
    # 'page-blocks.csv',
    # 'satimage.csv',
    # 'segment.csv',
    # 'winequality-red.csv',
    # 'winequality-white.csv',
    # 'contraceptive.csv',
    # 'newthyroid.csv',
    # 'tae.csv',

    'yeast.csv', # nie dziala dla gaussa z jakiegos powodu z baggingiem
    # 'abalone.csv' # ta sama sytuacja
]


def ensemble_tests(files, num_of_estimators, EnsembleClass, classifiers):
    alpha = .05
    n_folds = 5
    t_statistic = np.zeros((len(classifiers), len(classifiers)))
    p_value = np.zeros((len(classifiers), len(classifiers)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1234)

    for file in files:
        dataset = np.genfromtxt(f"new_datasets/{file}", delimiter=",")
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)
        scores = np.zeros((len(classifiers), n_folds))

        # divide dataset to folds
        for fold, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # ensemble learning
            for clf_index, (clf_name, clf) in enumerate(classifiers.items()):
                e_clf = EnsembleClass(base_estimator=clone(clf),
                                         n_estimators=num_of_estimators,
                                         random_state=0).fit(X_train, y_train)
                predict = e_clf.predict(X_test)
                scores[clf_index, fold] = accuracy_score(y_test, predict)

        # mean classifier scores
        print(f"\nDataset file: {file}")
        print(scores)
        mean = np.mean(scores, axis=1)
        std = np.std(scores, axis=1)

        for clf_id, (clf_name, clf) in enumerate(classifiers.items()):
            print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

        # pair statistical tests
        for i in range(len(classifiers)):
            for j in range(len(classifiers)):
                t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i],
                                                             scores[j])
        # print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
        headers = [key for key in classifiers.keys()]
        names_column = np.array([[key] for key in classifiers.keys()])
        t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
        # t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
        # p_value_table = np.concatenate((names_column, p_value), axis=1)
        # p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
        # print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
        advantage = np.zeros((len(classifiers), len(classifiers)))
        advantage[t_statistic > 0] = 1
        # advantage_table = tabulate(np.concatenate(
        #     (names_column, advantage), axis=1), headers)
        # print("Advantage:\n", advantage_table)
        significance = np.zeros((len(classifiers), len(classifiers)))
        significance[p_value <= alpha] = 1
        # significance_table = tabulate(np.concatenate(
        #     (names_column, significance), axis=1), headers)
        # print("Statistical significance (alpha = 0.05):\n", significance_table)
        stat_better = significance * advantage
        stat_better_table = tabulate(np.concatenate(
            (names_column, stat_better), axis=1), headers)
        print("Statistically significantly better:\n", stat_better_table)


bagging_clfs = {
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(),
    'MLP': MLPClassifier(),
}
print(f"**********\nBAGGING\n**********")
ensemble_tests(files=files, num_of_estimators=2, EnsembleClass=BaggingClassifier, classifiers=bagging_clfs)


adaboost_clfs = {
    'GNB': GaussianNB(),
    'CART': DecisionTreeClassifier(),
}
print(f"**********\nADABOOST\n**********")
ensemble_tests(files=files, num_of_estimators=2, EnsembleClass=AdaBoostClassifier, classifiers=adaboost_clfs)

