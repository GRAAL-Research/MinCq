#-*- coding:utf-8 -*-
__author__ = 'Jean-Francis Roy'

import numpy as np


class MajorityVote(object):
    """ A Majority Vote of real-valued functions.

    Parameters
    ----------
    voters : ndarray of Voter instances
        The voters of the majority vote. Each voter must take an example as an input, and output a real value in [-1,1].

    weights : ndarray, optional (default: uniform distribution)
        The weights associated to each voter.
    """
    def __init__(self, voters, weights=None):
        self._voters = np.array(voters)

        if weights is not None:
            assert(len(voters) == len(weights))
            self._weights = np.array(weights)
        else:
            self._weights = np.array([1.0 / len(voters)] * len(voters))

    def vote(self, X):
        """ Returns the vote of the Majority Vote on a list of samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify.

        Returns
        -------
        votes : ndarray, shape=(n_samples,), where each value is either -1 or 1
            The vote of the majority vote for each sample.
        """
        margins = self.margin(X)
        return np.array([int(x) for x in np.sign(margins)])

    def margin(self, X):
        """ Returns the margin of the Majority Vote on a list of samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data on which to calculate the margin.

        Returns
        -------
        margins : ndarray, shape=(n_samples,), where each value is either -1 or 1
            The margin of the majority vote for each sample.
        """
        classification_matrix = self.classification_matrix(X)
        return np.squeeze(np.asarray(np.dot(self.weights, classification_matrix)))


    def classification_matrix(self, X):
        """ Returns the classification matrix of the majority vote.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify

        Returns
        -------
        classification_matrix : ndrray, shape=(n_voters, n_samples)
            A matrix that contains the value output by each voter, for each sample.

        """
        return np.matrix([v.vote(X) for v in self._voters])

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = np.array(weights)

    @property
    def voters(self):
        return self._voters

    @voters.setter
    def voters(self, voters):
        self._voters = np.array(voters)
