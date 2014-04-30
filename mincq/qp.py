#-*- coding:utf-8 -*-
"""
This module encapsulates a quadratic program (QP) and solves it using CVXOPT.

It is not extremely well documented, please feel free to ask me for help if you have
any problem with it, or want to improve it.
"""
__author__ = 'Jean-Francis Roy'

import sys
import logging
import numpy as np


def _as_matrix(element):
    """ Utility function to convert anything to a Numpy matrix.
    """
    # If a scalar, return a 1x1 matrix.
    if len(np.shape(element)) == 0:
        return np.matrix([[element]], dtype=float)

    # If a nd-array vector, return a column matrix.
    elif len(np.shape(element)) == 1:
        matrix = np.matrix(element, dtype=float)
        if np.shape(matrix)[1] != 1:
            matrix = matrix.T
        return matrix

    return np.matrix(element, dtype=float)


def _as_column_matrix(array_like):
    """ Utility function to convert any array to a column Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[0] == 1:
        matrix = matrix.T

    return matrix


def _as_line_matrix(array_like):
    """ Utility function to convert any array to a line Numpy matrix.
    """
    matrix = _as_matrix(array_like)
    if 1 not in np.shape(matrix):
        raise ValueError("_as_column_vector: input must be a vector")

    if np.shape(matrix)[1] == 1:
        matrix = matrix.T

    return matrix


class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   http://www.electricmonk.nl/log/2011/08/14/redirect-stdout-and-stderr-to-a-logger-in-python/
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())


class QP(object):
    """
    Encapsulates a quadratic program of the following form:

    minimize    (1/2)*x'*P*x + q'*x
    subject to  G*x <= h
                A*x = b.
    """
    def __init__(self):
        self._quadratic_func = None
        self._linear_func = None
        self._inequality_constraints_matrix = None
        self._inequality_constraints_values = None
        self._equality_constraints_matrix = None
        self._equality_constraints_values = None
        self._lower_bound_values = None
        self._upper_bound_values = None
        self._n_variables = 0

    @property
    def n_variables(self):
        return self._n_variables

    @property
    def quadratic_func(self):
        return self._quadratic_func

    @quadratic_func.setter
    def quadratic_func(self, quad_matrix):
        quad_matrix = _as_matrix(quad_matrix)
        n_lines, n_columns = np.shape(quad_matrix)
        assert(n_lines == n_columns)

        self._n_variables = n_lines
        self._quadratic_func = quad_matrix

    @property
    def linear_func(self):
        return self._linear_func

    @linear_func.setter
    def linear_func(self, lin_vector):
        self._assert_quadratic_func_is_set()

        lin_vector = _as_column_matrix(lin_vector)
        assert(np.shape(lin_vector)[0] == self._n_variables)

        self._linear_func = lin_vector

    def add_inequality_constraints(self, inequality_matrix, inequality_values):
        if inequality_matrix is None:
            print("Empty inequality constraint: ignoring!")
            return

        self._assert_quadratic_func_is_set()
        if 1 in np.shape(inequality_matrix) or len(np.shape(inequality_matrix)) == 1:
            inequality_matrix = _as_line_matrix(inequality_matrix)
        else:
            inequality_matrix = _as_matrix(inequality_matrix)

        inequality_values = _as_matrix(inequality_values)
        assert(np.shape(inequality_matrix)[1] == self._n_variables)
        assert(np.shape(inequality_values)[1] == 1)

        if self._inequality_constraints_matrix is None:
            self._inequality_constraints_matrix = inequality_matrix
        else:
            self._inequality_constraints_matrix = np.append(self._inequality_constraints_matrix,
                                                            inequality_matrix, axis=0)

        if self._inequality_constraints_values is None:
            self._inequality_constraints_values = inequality_values
        else:
            self._inequality_constraints_values = np.append(self._inequality_constraints_values,
                                                            inequality_values, axis=0)

    def add_equality_constraints(self, equality_matrix, equality_values):
        if equality_matrix is None:
            print("Empty equality constraint: ignoring!")
            return

        self._assert_quadratic_func_is_set()
        if 1 in np.shape(equality_matrix) or len(np.shape(equality_matrix)) == 1:
            equality_matrix = _as_line_matrix(equality_matrix)
        else:
            equality_matrix = _as_matrix(equality_matrix)

        equality_values = _as_matrix(equality_values)
        assert(np.shape(equality_matrix)[1] == self._n_variables)
        assert(np.shape(equality_values)[1] == 1)

        if self._equality_constraints_matrix is None:
            self._equality_constraints_matrix = equality_matrix
        else:
            self._equality_constraints_matrix = np.append(self._equality_constraints_matrix,
                                                          equality_matrix, axis=0)

        if self._equality_constraints_values is None:
            self._equality_constraints_values = equality_values
        else:
            self._equality_constraints_values = np.append(self._equality_constraints_values,
                                                          equality_values, axis=0)

    def add_lower_bound(self, lower_bound):
        self._lower_bound_values = np.array([lower_bound] * self.n_variables)

    def add_upper_bound(self, upper_bound):
        self._upper_bound_values = np.array([upper_bound] * self.n_variables)

    def _convert_bounds_to_inequality_constraints(self):
        if self._lower_bound_values is not None:
            c_matrix = []
            for i in range(self.n_variables):
                c_line = [0] * self.n_variables
                c_line[i] = -1.0
                c_matrix.append(c_line)

            c_vector = _as_column_matrix(self._lower_bound_values)
            self._lower_bound_values = None
            self.add_inequality_constraints(np.matrix(c_matrix).T, c_vector)

        if self._upper_bound_values is not None:
            c_matrix = []
            for i in range(self.n_variables):
                c_line = [0] * self.n_variables
                c_line[i] = 1.0
                c_matrix.append(c_line)

            c_vector = _as_column_matrix(self._upper_bound_values)
            self._upper_bound_values = None
            self.add_inequality_constraints(np.matrix(c_matrix).T, c_vector)

    def _assert_quadratic_func_is_set(self):
        assert self.quadratic_func is not None, "Please set quadratic_func first"

    def _convert_to_cvxopt_matrices(self):
        from cvxopt import matrix as cvxopt_matrix
        self._quadratic_func = cvxopt_matrix(self._quadratic_func)

        if self._linear_func is not None:
            self._linear_func = cvxopt_matrix(self._linear_func)
        else:
            # CVXOPT needs this vector to be set even if it is not used, so we put zeros in it!
            self._linear_func = cvxopt_matrix(np.zeros((self._n_variables, 1)))

        if self._inequality_constraints_matrix is not None:
            self._inequality_constraints_matrix = cvxopt_matrix(self._inequality_constraints_matrix)

        if self._inequality_constraints_values is not None:
            self._inequality_constraints_values = cvxopt_matrix(self._inequality_constraints_values)

        if self._equality_constraints_matrix is not None:
            self._equality_constraints_matrix = cvxopt_matrix(self._equality_constraints_matrix)

        if self._equality_constraints_values is not None:
            self._equality_constraints_values = cvxopt_matrix(self._equality_constraints_values)

    def solve(self, feastol=1e-7, abstol=1e-7, reltol=1e-6):
        from cvxopt.solvers import qp, options
        options['feastol'] = feastol
        options['abstol'] = abstol
        options['reltol'] = reltol

        # CVXOPT is very verbose, and we don't want it to pollute STDOUT or STDERR

        original_stdout = sys.stdout
        original_stderr = sys.stderr

        stdout_logger = logging.getLogger('CVXOPT')
        sl = StreamToLogger(stdout_logger, logging.INFO)
        sys.stdout = sl

        stderr_logger = logging.getLogger('CVXOPT')
        sl = StreamToLogger(stderr_logger, logging.ERROR)
        sys.stderr = sl

        self._convert_bounds_to_inequality_constraints()
        self._convert_to_cvxopt_matrices()

        try:
            ret = qp(self.quadratic_func, self.linear_func, self._inequality_constraints_matrix,
                     self._inequality_constraints_values, self._equality_constraints_matrix,
                     self._equality_constraints_values)
            ret = np.asarray(np.array(ret['x']).T[0])
        except:
            raise

        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        return ret
