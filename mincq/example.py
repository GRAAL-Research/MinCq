#-*- coding:utf-8 -*-
"""MinCq learning algorithm usage examples

Related papers:

[1] F. Laviolette, M. Marchand, J.-F. Roy, "From PAC-Bayes Bounds to Quadratic Programs for Majority Votes",
    In Proceedings of the 28th International Conference on Machine Learning, Bellevue, WA, USA, June 2011.

[2] P. Germain, A. Lacasse, F. Laviolette, M. Marchand, J.-F. Roy, "Risk Bounds for the Majority Vote: From a
    PAC-Bayesian Analysis to a Learning Algorithm", accepted for publication in the Journal of Machine Learning
    Research.

http://graal.ift.ulaval.ca/majorityvote/
"""
__author__ = 'Jean-Francis Roy'

import logging
import pylab as pl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import zero_one_loss
from sklearn.metrics.scorer import accuracy_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from matplotlib.colors import ListedColormap

from utils import print_sklearn_grid_scores
import voter


def main():
    # Change logging.ERROR to logging.INFO to activate more verbose information.
    logging.basicConfig(level=logging.INFO)

    # Four simple examples of MinCq usage.
    simple_classification_example()
    multi_voters_example()
    cross_validation_example()
    scikit_learn_classifier_comparison_example()


def simple_classification_example():
    """ Simple example : with fixed hyperparameters, run four versions of MinCq on a single dataset.
    """
    # MinCq parameters, fixed to a given value as this is a simple example.
    mu = 0.001

    # We load iris dataset, We convert the labels to be -1 or 1, and we split it in two parts: train and test.
    dataset = load_iris()
    dataset.target[dataset.target == 0] = -1
    dataset.target[dataset.target == 2] = -1
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42)

    # We train MinCq using decision stumps as voters, on the training set.
    learner = MinCqLearner(mu, voters_type='stumps')
    learner.fit(X_train, y_train)

    # We predict the train and test labels and print the risk.
    predictions_train = learner.predict(X_train)
    predictions_test = learner.predict(X_test)

    print("\nStumpsMinCq")
    print("-----------")
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}\n".format(zero_one_loss(y_test, predictions_test)))

    # We do the same again, now with a linear kernel.
    learner = MinCqLearner(mu, voters_type='kernel', kernel='linear')
    learner.fit(X_train, y_train)

    predictions_train = learner.predict(X_train)
    predictions_test = learner.predict(X_test)

    print("\nLinearMinCq")
    print("-----------")
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}\n".format(zero_one_loss(y_test, predictions_test)))

    # We do the same again, now with a polynomial kernel.
    learner = MinCqLearner(mu, voters_type='kernel', kernel='poly')
    learner.fit(X_train, y_train)

    predictions_train = learner.predict(X_train)
    predictions_test = learner.predict(X_test)

    print("\nPolyMinCq")
    print("-----------")
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}\n".format(zero_one_loss(y_test, predictions_test)))

    # We do the same again, now with an RBF kernel.
    learner = MinCqLearner(mu, voters_type='kernel', kernel='rbf', gamma=0.0)
    learner.fit(X_train, y_train)

    predictions_train = learner.predict(X_train)
    predictions_test = learner.predict(X_test)

    print("\nRbfMinCq")
    print("--------")
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}\n".format(zero_one_loss(y_test, predictions_test)))


def multi_voters_example():
    """ Example of using a combination of many types of voters, which may be seen as multi-kernel learning (MKL).

    This particular dataset is easy to solve and combining voters degrades performance. However, it might be a good
    idea for a more complex dataset.
    """
    # MinCq parameters, fixed to a given value as this is a simple example.
    mu = 0.001

    # We load iris dataset, We convert the labels to be -1 or 1, and we split it in two parts: train and test.
    dataset = load_iris()
    dataset.target[dataset.target == 0] = -1
    dataset.target[dataset.target == 2] = -1
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42)

    # We create a set of voters of different kind.
    voters = voter.StumpsVotersGenerator(10).generate(X_train)
    voters = np.append(voters, voter.KernelVotersGenerator(rbf_kernel, gamma=0.01).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(rbf_kernel, gamma=0.1).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(rbf_kernel, gamma=1).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(rbf_kernel, gamma=10).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(rbf_kernel, gamma=100).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(polynomial_kernel, degree=2).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(polynomial_kernel, degree=3).generate(X_train))
    voters = np.append(voters, voter.KernelVotersGenerator(linear_kernel).generate(X_train))

    # We train MinCq using these voters, on the training set.
    learner = MinCqLearner(mu, voters_type='manual')
    learner.voters = voters
    learner.fit(X_train, y_train)

    # We predict the train and test labels and print the risk.
    predictions_train = learner.predict(X_train)
    predictions_test = learner.predict(X_test)

    print("\nMultiVotersMinCq")
    print("-----------")
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}\n".format(zero_one_loss(y_test, predictions_test)))


def cross_validation_example():
    """ Slightly more complex example : Perform grid search cross-validation to find optimal parameters for MinCq using
    rbf kernels as voters.
    """
    # We load iris dataset, We convert the labels to be -1 or 1, and we split it in two parts: train and test.
    dataset = load_iris()
    dataset.target[dataset.target == 0] = -1
    dataset.target[dataset.target == 2] = -1
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=42)

    # The learning algorithm and its parameters.
    learner = MinCqLearner(mu=0.0001, voters_type='kernel', kernel='rbf', gamma=0.0)
    learner_params = {'mu': [0.0001, 0.001, 0.01],
                      'gamma': [0.0, 0.1, 1.0, 10]}

    cv_classifier = GridSearchCV(learner, learner_params, scoring=accuracy_scorer)
    cv_classifier = cv_classifier.fit(X_train, y_train)

    predictions_train = cv_classifier.predict(X_train)
    predictions_test = cv_classifier.predict(X_test)

    print_sklearn_grid_scores("Iris", "RbfMinCq", learner_params, cv_classifier.grid_scores_)

    print("Best parameters: {}".format(str(cv_classifier.best_params_)))
    print("Training set risk: {:.4f}".format(zero_one_loss(y_train, predictions_train)))
    print("Testing set risk: {:.4f}".format(zero_one_loss(y_test, predictions_test)))


def scikit_learn_classifier_comparison_example():
    """ Example of usage for MinCq learning algorithm, based on Scikit-Learn's Classifier comparison example.
    Original source of the example: http://scikit-learn.org/stable/auto_examples/plot_classifier_comparison.html

    Please note that SVM and AdaBoost results from these papers may differ from this implementation: Scikit-Learn
    implementations were not used for all the experiments of the papers.
    """

    # Code source: Gael Varoqueux
    #              Andreas Mueller
    # Modified for Documentation merge by Jaques Grobler
    # Modified to serve as a MinCq example by Jean-Francis Roy
    # License: BSD 3 clause

    h = .02  # step size in the mesh

    names = ["Linear SVM", "RBF SVM", "AdaBoost", "Linear MinCq", "RBF MinCq", "Stumps MinCq"]
    classifiers = [
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        AdaBoostClassifier(),
        MinCqLearner(mu=0.01, voters_type="kernel", kernel="linear"),
        MinCqLearner(mu=0.01, voters_type="kernel", kernel="rbf", gamma=2),
        MinCqLearner(mu=0.01, voters_type="stumps"),
    ]

    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=1, n_clusters_per_class=1)

    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [make_moons(noise=0.3, random_state=0),
                make_circles(noise=0.2, factor=0.5, random_state=1),
                linearly_separable
                ]

    figure = pl.figure(figsize=(27, 9))
    i = 1
    # iterate over datasets
    for ds in datasets:
        # preprocess dataset, split into training and test part
        X, y = ds
        y[y == 0] = -1
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = pl.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = pl.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

    figure.subplots_adjust(left=.02, right=.98)
    pl.show()


if __name__ == '__main__':
    main()
