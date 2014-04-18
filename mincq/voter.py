#-*- coding:utf-8 -*-
__author__ = "Jean-Francis Roy"

import numpy as np


class Voter(object):
    """ Base class for a voter (function X -> [-1, 1]), where X is an array of samples
    """
    def __init__(self):
        pass

    def vote(self, X):
        """ Returns the output of the voter, on a sample list X

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data to classify

        Returns
        -------
        votes : ndarray, shape=(n_samples,)
            The result the the voter function, for each sample
        """
        raise NotImplementedError("Voter.vote: Not implemented.")


class BinaryKernelVoter(Voter):
    """ A Binary Kernel Voter, which outputs the value of a kernel function whose first example is fixed a priori.
    The sign of the output depends on the label (-1 or 1) of the sample on which the kernel voter is based

    Parameters
    ----------
    x : ndarray, shape=(n_features,)
        The base sample's description vector

    y : int, -1 or 1
        The label of the base sample. Determines if the voter thinks "negative" or "positive"

    kernel_function : function
        The kernel function takes two samples and returns a similarity value. If the kernel has parameters, they should
        be set using kwargs parameter

    kwargs : keyword arguments (optional)
        Additional parameters for the kernel function
    """

    def __init__(self, x, y, kernel_function, **kwargs):
        assert(y in {-1, 1})
        super(BinaryKernelVoter, self).__init__()
        self._x = x
        self._y = y
        self._kernel_function = kernel_function
        self._kernel_kwargs = kwargs

    def vote(self, X):
        base_point_array = np.array([self._x])
        votes = self._y * self._kernel_function(base_point_array, X, **self._kernel_kwargs)
        votes = np.squeeze(np.asarray(votes))

        return votes


class DecisionStumpVoter(Voter):
    """
    Generic Attribute Threshold Binary Classifier

    Parameters
    ----------
    attribute_index : int
        The attribute to consider for the classification

    threshold : float
        The threshold value for classification rule

    direction : int (-1 or 1)
        Used to reverse classification decision
    """
    def __init__(self, attribute_index, threshold, direction=1):
        super(DecisionStumpVoter, self).__init__()
        self.attribute_index = attribute_index
        self.threshold = threshold
        self.direction = direction

    def vote(self, points):
        return [((point[self.attribute_index] > self.threshold) * 2 - 1) * self.direction for point in points]


class VotersGenerator(object):
    """ Base class to create a set of voters using training samples
    """

    def generate(self, X, y=None, self_complemented=False):
        """ Generates the voters using samples.

        Parameters
        ----------
        X : ndarray, shape=(n_samples, n_features)
            Input data on which to base the voters

        y : ndarray, shape=(n_samples,), optional
            Input labels, usually determines the decision polarity of each voter

        self_complemented : bool
            Determines if complement voters should be generated or not

        Returns
        -------
        voters : ndarray
            An array of voters
        """
        raise NotImplementedError("VotersGenerator.generate: not implemented")


class StumpsVotersGenerator(VotersGenerator):
    """ Decision Stumps Voters generator.

    Parameters
    ----------
    n_stumps_per_attribute : int, (default=10)
        Determines how many decision stumps will be created for each attribute.
    """
    def __init__(self, n_stumps_per_attribute=10):
        self._n_stumps_per_attribute = n_stumps_per_attribute

    def _find_extremums(self, X, i):
        mini = np.Infinity
        maxi = -np.Infinity
        for x in X:
            if x[i] < mini:
                mini = x[i]
            if x[i] > maxi:
                maxi = x[i]
        return mini, maxi

    def generate(self, X, y=None, self_complemented=False):
        voters = []
        if len(X) != 0:
            for i in range(len(X[0])):
                t = self._find_extremums(X, i)
                inter = (t[1] - t[0]) / (self._n_stumps_per_attribute + 1)

                if inter != 0:
                    # If inter is zero, the attribute is useless as it has a constant value. We do not add stumps for
                    # this attribute.
                    for x in range(self._n_stumps_per_attribute):
                        voters.append(DecisionStumpVoter(i, t[0] + inter * (x + 1), 1))

                        if self_complemented:
                            voters.append(DecisionStumpVoter(i, t[0] + inter * (x + 1), -1))

        return np.array(voters)


class KernelVotersGenerator(VotersGenerator):
    """ Utility function to create binary kernel voters for each (x, y) sample.

    Parameters
    ----------
    kernel_function : function
        The kernel function takes two samples and returns a similarity value. If the kernel has parameters, they should
        be set using kwargs parameter

    kwargs : keyword arguments (optional)
        Additional parameters for the kernel function
    """

    def __init__(self, kernel_function, **kwargs):
        self._kernel_function = kernel_function
        self._kernel_kwargs = kwargs

    def generate(self, X, y=None, self_complemented=False):
        if y is None:
            y = 1

        voters = []
        for point, label in zip(X, y):
            voters.append(BinaryKernelVoter(point, label, self._kernel_function, **self._kernel_kwargs))

        if self_complemented:
            for point, label in zip(X, y):
                voters.append(BinaryKernelVoter(point, -1 * label, self._kernel_function, **self._kernel_kwargs))

        return np.array(voters)
