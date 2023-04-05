##
## Ann Bonde
## ID: 5423260
## CSCI 5521 Homework 2, Question 1
## To run, call MultiGaussian(training file name, testing file name, model #)
## prints found parameters and error rates
## returns parameters

import numpy as np


def model1(train_x, train_y, test_x, test_y):
    # split data set on class
    train_c1 = train_x[np.where(train_y[:, 0] == 1)]
    train_c2 = train_x[np.where(train_y[:, 0] == 2)]

    # get priors from data
    p_c1 = np.count_nonzero(train_y == 1) / len(train_y)
    p_c2 = np.count_nonzero(train_y == 2) / len(train_y)

    # get means and cov matrix for c1 and c2
    u1 = np.mean(train_c1, axis=0)
    u2 = np.mean(train_c2, axis=0)
    s1 = np.cov(np.transpose(train_c1))
    s2 = np.cov(np.transpose(train_c2))
    print(f"Model 1 Parameters (rounded)\nu1: {np.around(u1, 2)}\nu2: {np.around(u2, 2)}\nS1: {np.around(s1, 2)}\nS2: {np.around(s2, 2)}")

    # empty array for prediction
    pred_y = np.zeros(test_y.shape)
    # compute terms of discriminant function using eqn 5.19 from the book
    # these are constant for each x
    term1_c1 = -0.5 * np.log(np.linalg.det(s1))
    term1_c2 = -0.5 * np.log(np.linalg.det(s2))
    term3_c1 = np.log(p_c1)
    term3_c2 = np.log(p_c2)
    for i in range(test_x.shape[0]):
        # term2: 0.5* xT*Si-1*x - 2xT*Si-1*mi + miT*Si-1*mi
        term2_c1 = - 0.5 * (np.matmul(np.transpose(test_x[i]), np.matmul(np.linalg.inv(s1), test_x[i]))
                            - 2 * (np.matmul(np.transpose(test_x[i, :]), np.matmul(np.linalg.inv(s1), u1)))
                            + np.matmul(np.transpose(u1), np.matmul(np.linalg.inv(s1), u1)))
        term2_c2 = - 0.5 * (np.matmul(np.transpose(test_x[i]), np.matmul(np.linalg.inv(s2), test_x[i]))
                            - 2 * (np.matmul(np.transpose(test_x[i]), np.matmul(np.linalg.inv(s2), u2)))
                            + np.matmul(np.transpose(u2), np.matmul(np.linalg.inv(s2), u2)))
        # add all terms
        g1 = term1_c1 + term2_c1 + term3_c1
        g2 = term1_c2 + term2_c2 + term3_c2
        # log likelihood
        if np.log(g1 / g2) < 0:
            pred_y[i] = 1
        else:
            pred_y[i] = 2
    # find error on test set
    err = 1 - np.mean(pred_y == test_y)
    err_c1 = 1-np.mean(test_y[np.where(test_y == 1)] == pred_y[np.where(test_y == 1)])
    err_c2 = 1-np.mean(test_y[np.where(test_y == 2)] == pred_y[np.where(test_y == 2)])
    print(f"Model 1 Total Error: {err}\n\tC1 Error: {err_c1}\n\tC2 Error: {err_c2}")


    # return u1, u2, S1, s2
    return u1, u2, s1, s2


def model2(train_x, train_y, test_x, test_y):
    # split data set on class
    train_c1 = train_x[np.where(train_y[:, 0] == 1)]
    train_c2 = train_x[np.where(train_y[:, 0] == 2)]

    # get priors from data
    p_c1 = np.count_nonzero(train_y == 1) / len(train_y)
    p_c2 = np.count_nonzero(train_y == 2) / len(train_y)

    # find mean and covar matrix
    u1 = np.mean(train_c1, axis=0)
    u2 = np.mean(train_c2, axis=0)
    s1 = np.cov(np.transpose(train_c1))
    s2 = np.cov(np.transpose(train_c2))
    # from eqn 5.21 in book: s = sum(p(Ci)Si)
    s = p_c1 * s1 + p_c2 * s2
    print(f"Model 2 Parameters (rounded)\nu1: {np.around(u1, 2)}\nu2: {np.around(u2, 2)}\nS1 = S2: {np.around(s, 2)}")

    # empty array for prediction
    pred_y = np.zeros(test_y.shape)
    for i in range(test_x.shape[0]):
        g1 = -0.5 * np.matmul(np.transpose(test_x[i] - u1), np.matmul(np.linalg.inv(s), test_x[i] - u1)) + np.log(p_c1)
        g2 = -0.5 * np.matmul(np.transpose(test_x[i] - u2), np.matmul(np.linalg.inv(s), test_x[i] - u2)) + np.log(p_c2)
        # log likelihood
        if np.log(g1 / g2) < 0:
            pred_y[i] = 1
        else:
            pred_y[i] = 2
    # find error on test set
    err = 1 - np.mean(pred_y == test_y)
    err_c1 = 1 - np.mean(test_y[np.where(test_y == 1)] == pred_y[np.where(test_y == 1)])
    err_c2 = 1 - np.mean(test_y[np.where(test_y == 2)] == pred_y[np.where(test_y == 2)])
    print(f"Model 2 Total Error: {err}\n\tC1 Error: {err_c1}\n\tC2 Error: {err_c2}")

    return u1, u2, s


def model3(train_x, train_y, test_x, test_y):
    # split data set on class
    train_c1 = train_x[np.where(train_y[:, 0] == 1)]
    train_c2 = train_x[np.where(train_y[:, 0] == 2)]

    # get priors from data
    p_c1 = np.count_nonzero(train_y == 1) / len(train_y)
    p_c2 = np.count_nonzero(train_y == 2) / len(train_y)

    # find mean and covar matrix
    u1 = np.mean(train_c1, axis=0)
    u2 = np.mean(train_c2, axis=0)
    #take diagonals of covariance matrix
    s1 = np.diag(np.cov(np.transpose(train_c1)))
    s2 = np.diag(np.cov(np.transpose(train_c2)))
    print(f"Model 3 Parameters (rounded)\nu1: {np.around(u1, 2)}\nu2: {np.around(u2, 2)}\n\u03C31^2: {np.around(s1, 2)}\n\u03C32^2: {np.around(s2, 2)}")

    # empty array for prediction
    pred_y = np.zeros(test_y.shape)
    g1 = 0
    g2 = 0
    for i in range(test_x.shape[0]):
        #from eqn 5.24 in book
        for j in range(test_x.shape[1]):
            g1 = g1 + ((test_x[i, j] - u1[j]) / s1[j]) ** 2
            g2 = g2 + ((test_x[i, j] - u2[j]) / s2[j]) ** 2
        g1 = -0.5 * g1 + np.log(p_c1)
        g2 = -0.5 * g2 + np.log(p_c2)
        # log likelihood
        if np.log(g1 / g2) < 0:
            pred_y[i] = 1
        else:
            pred_y[i] = 2
        g1 = 0
        g2 = 0
    # find error on test set
    err = 1 - np.mean(pred_y == test_y)
    err_c1 = 1 - np.mean(test_y[np.where(test_y == 1)] == pred_y[np.where(test_y == 1)])
    err_c2 = 1 - np.mean(test_y[np.where(test_y == 2)] == pred_y[np.where(test_y == 2)])
    print(f"Model 3 Total Error: {err}\n\tC1 Error: {err_c1}\n\tC2 Error: {err_c2}")

    return u1, u2, s1, s2


def MultiGaussian(training_fname, testing_fname, model = 0):
    train = np.loadtxt(training_fname, delimiter=',')
    test = np.loadtxt(testing_fname, delimiter=',')
    train_x = train[:, :-1]
    train_y = train[:, -1:]
    test_x = test[:, :-1]
    test_y = test[:, -1:]

    if model ==0:
        model1(train_x, train_y, test_x, test_y)
        model2(train_x, train_y, test_x, test_y)
        model3(train_x, train_y, test_x, test_y)
    if model ==1:
        model1(train_x, train_y, test_x, test_y)
    if model ==2:
        model3(train_x, train_y, test_x, test_y)
    if model ==3:
        model3(train_x, train_y, test_x, test_y)


