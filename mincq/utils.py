#-*- coding:utf-8 -*-
""" Module containing small utilities that have nowhere else to go. """
__author__ = 'Jean-Francis Roy'

import numpy as np

def print_sklearn_grid_scores(dataset_name, algorithm_name, param_grid, grid_scores):
    """ This function does a prettier print of a grid of parameters and scores, used in Scikit-Learn's GridSearchCV.
    Only works for 1 or 2 parameters.

    Parameters
    ----------
    dataset_name : string
        The name of the dataset.

    algorithm_name : string
        The name of the algorithm.

    param_grid: dict
        The parameter grid, using the same format than Scikit-Learn's GridSearchCV.

    grid_scores: list of named tuples
        The grid score, use GridSearchCV's grid_scores_ attribute.
    """
    n_params = len(param_grid.keys())
    assert n_params <= 2, "print_sklearn_grid_scores: Only works with 1 or 2 parameters."

    x_param_name, x_param_values = list(param_grid.items())[0]
    x_param_values = np.array(x_param_values)
    n_x_param_values = len(x_param_values)

    n_y_param_values = 0
    if n_params == 2:
        y_param_name, y_param_values = list(param_grid.items())[1]
        y_param_values = np.array(y_param_values)
        n_y_param_values = len(y_param_values)


    if n_params == 1:
        mean_risks = np.zeros(n_x_param_values)
    elif n_params == 2:
        mean_risks = np.zeros((n_y_param_values, n_x_param_values))

    for score in grid_scores:
        if n_params == 1:
            mean_risks[np.where(x_param_values == score.parameters[x_param_name])[0][0]] = 1 - score.mean_validation_score
        elif n_params == 2:
            mean_risks[np.where(y_param_values == score.parameters[y_param_name])[0][0]][np.where(x_param_values == score.parameters[x_param_name])[0][0]] = 1 - score.mean_validation_score


    print("+" + "-"*(17 + 9*n_x_param_values) + "+")
    print(("|{:^" + str(17 + 9*n_x_param_values) + "}|").format("%s on %s" % (algorithm_name, dataset_name)))
    print("+" + "-"*17 + ("+" + "-"*8) * n_x_param_values + "+")
    print(("|{:14}-> |" + "{:8.4f}|"*n_x_param_values).format(x_param_name, *x_param_values))

    if n_params == 1:
        print(("|{:14}   |" + "{:8.4f}|"*n_x_param_values).format("", *mean_risks))
    elif n_params == 2:
        print(("|{:14}   |" + "{:8}|"*n_x_param_values).format(y_param_name, *[""]*n_x_param_values))
        for i in range(n_y_param_values):
            print(("|{:14}   |" + "{:8.4f}|"*n_x_param_values).format(y_param_values[i], *mean_risks[i]))
    print("+" + "-"*17 + ("+" + "-"*8) * n_x_param_values + "+")
    print("")
