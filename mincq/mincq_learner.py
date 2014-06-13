#-*- coding:utf-8 -*-
""" MinCq learning algorithm

Related papers:
[1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes (Laviolette et al., 2011)
[2] Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm (Germain et al., 2014)

http://graal.ift.ulaval.ca/majorityvote/
"""
__author__ = 'Jean-Francis Roy'

import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from qp import QP
from majority_vote import MajorityVote
from voter import StumpsVotersGenerator, KernelVotersGenerator



class MinCqLearner(BaseEstimator, ClassifierMixin):
    """
    MinCq algorithm learner. See [1, 2]

    Parameters
    ----------
    mu : float
        The fixed value of the first moment of the margin.

    voters_type : string, optional (default='kernel')
        Specifies the type of voters.
        It must be one of 'kernel', 'stumps' or 'manual'. If 'manual' is specified, the voters have to be manually set
        using the "voters" parameter of the fit function.

    n_stumps_per_attribute : int, optional (default=10)
        Specifies the amount of decision stumps per attribute.
        It is only significant with 'stumps' voters_type.

    kernel : string, optional (default='rbf')
         Specifies the kernel type to be used in the algorithm.
         It must be one of 'linear', 'poly', 'rbf'.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=0.0)
        Kernel coefficient for 'rbf' and 'poly'.
        If gamma is 0.0 then 1/n_features will be used instead.
    """
    def __init__(self, mu, voters_type, n_stumps_per_attribute=10, kernel='rbf', degree=3, gamma=0.0):
        assert mu > 0 and mu <= 1, "MinCqLearner: mu parameter must be in (0, 1]"
        self.mu = mu
        self.voters_type = voters_type
        self.n_stumps_per_attribute = n_stumps_per_attribute
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma

        self.majority_vote = None
        self.qp = None

    def fit(self, X, y, voters=None):
        """ Learn a majority vote weights using MinCq.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Training data

        y : ndarray, shape=(n_samples,), optional
            Training labels

        voters : shape=(n_voters,), optional
            A priori generated voters
        """
        # Preparation of the majority vote, using a voter generator that depends on class attributes

        assert self.voters_type in ['stumps', 'kernel', 'manual'], "MinCqLearner: voters_type must be 'stumps', 'kernel' or 'manual'"

        if self.voters_type == 'manual':
            if voters is None:
                logging.error("Manually set voters is True, but no voters have been set.")
                return self

        else:
            voters_generator = None

            if self.voters_type == 'stumps':
                assert self.n_stumps_per_attribute > 1, 'MinCqLearner: n_stumps_per_attribute must be positive'
                voters_generator = StumpsVotersGenerator(self.n_stumps_per_attribute)

            elif self.voters_type == 'kernel':
                assert self.kernel in ['linear', 'poly', 'rbf'], "MinCqLearner: kernel must be 'linear', 'poly' or 'rbf'"

                gamma = self.gamma
                if gamma == 0.0:
                    gamma = 1.0 / np.shape(X)[1]

                if self.kernel == 'linear':
                    voters_generator = KernelVotersGenerator(linear_kernel)
                elif self.kernel == 'poly':
                    voters_generator = KernelVotersGenerator(polynomial_kernel, degree=self.degree, gamma=gamma)
                elif self.kernel == 'rbf':
                    voters_generator = KernelVotersGenerator(rbf_kernel, gamma=gamma)

            voters = voters_generator.generate(X, y)

        logging.info("MinCq training started...")
        logging.info("Training dataset shape: {}".format(str(np.shape(X))))
        logging.info("Number of voters: {}".format(len(voters)))

        self.majority_vote = MajorityVote(voters)
        n_base_voters = len(self.majority_vote.weights)

        # Preparation and resolution of the quadratic program
        logging.info("Preparing QP...")
        self._prepare_qp(X, y)

        try:
            logging.info("Solving QP...")
            solver_weights = self.qp.solve()

            # Conversion of the weights of the n first voters to weights on the implicit 2n voters.
            # See Section 7.1 of [2] for an explanation.
            self.majority_vote.weights = np.array([2 * q - 1.0 / n_base_voters for q in solver_weights])
            logging.info("First moment of the margin on the training set: {:.4f}".format(np.mean(y * self.majority_vote.margin(X))))

        except Exception as e:
            logging.error("{}: Error while solving the quadratic program: {}.".format(str(self), str(e)))
            self.majority_vote = None

        return self

    def predict(self, X):
        """ Using previously learned majority vote weights, predict the labels of new data points.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Samples to predict

        Returns
        -------
        predictions : ndarray, shape=(n_samples,)
            The predicted labels
        """
        logging.info("Predicting...")
        if self.majority_vote is None:
            logging.error("{}: Error while predicting: MinCq has not been fit or fitting has failed. Will output invalid labels".format(str(self)))
            return np.zeros((len(X),))

        return self.majority_vote.vote(X)

    def predict_proba(self, X):
        """ Using previously learned majority vote weights, predict the labels of new data points with a confidence
        level. The confidence level is the margin of the majority vote.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Samples to predict

        Returns
        -------
        predictions : ndarray, shape=(n_samples,)
            The predicted labels
        """
        probabilities = np.zeros((np.shape(X)[0], 2))

        # The margin is between -1 and 1, we rescale it to be between 0 and 1.
        margins = self.majority_vote.margin(X)
        margins += 1
        margins /= 2

        # Then, the conficence for class +1 is set to the margin, and confidence for class -1 is set to 1 - margin.
        probabilities[:, 1] = margins
        probabilities[:, 0] = 1 - margins
        return probabilities

    def _prepare_qp(self, X, y):
        """ Prepare MinCq's quadratic program. See Program 1 of [2] for more details on its content.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Training data

        y : ndarray, shape=(n_samples,), optional
            Training labels
        """

        self.qp = QP()

        n_features = len(self.majority_vote.voters)
        classification_matrix = self.majority_vote.classification_matrix(X)

        # Objective function
        c_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                c_matrix[i][j] = np.mean(np.multiply(classification_matrix[i], classification_matrix[j]))
        self.qp.quadratic_func = 2 * np.matrix(c_matrix)
        self.qp.linear_func = np.matrix(np.matrix(-1.0 * np.mean(self.qp.quadratic_func / 2.0, axis=1))).T

        # First moment of the margin fixed to mu.
        c_matrix = np.zeros((self.qp.n_variables,))
        for i in range(self.qp.n_variables):
            c_matrix[i] = np.mean(np.multiply(np.asarray(classification_matrix)[i], y))

        column_means = np.mean(np.multiply(y, classification_matrix), axis=1)
        self.qp.add_equality_constraints(c_matrix, 0.5 * (self.mu + np.mean(column_means)))

        # Lower and upper bounds on the variables
        self.qp.add_lower_bound(0.0)
        self.qp.add_upper_bound(1.0 / n_features)

