# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:05:24 2013

@author: vguigue
"""

import matplotlib.pyplot as pl
import numpy as np


def dataset(dataType="gaussian", napp=1000, ntest=1000, sigma=0.3):
    if dataType == "gaussian":
        mu = 2
        x = np.vstack(
            (
                np.random.randn(napp, 2) * sigma + mu,
                np.random.randn(napp, 2) * sigma - mu,
            )
        )
        y = np.vstack((np.ones((napp, 1)), np.ones((napp, 1)) * -1))

        xtest = np.vstack(
            (
                np.random.randn(ntest, 2) * sigma + mu,
                np.random.randn(ntest, 2) * sigma - mu,
            )
        )
        ytest = np.vstack((np.ones((ntest, 1)), np.ones((ntest, 1)) * -1))
        return x, y, xtest, ytest

    elif dataType == "clowns":
        dilatation = 2.5
        offset = -1.5
        vdil = -offset
        x1 = np.random.randn(napp, 2) * sigma
        x1[:, 0] *= 2
        x2 = np.random.rand(napp, 2) - 0.5

        x2[:, 0] *= dilatation
        x2[:, 1] += offset + vdil * x2[:, 0] * x2[:, 0]
        x2[:, 1:2] += np.random.randn(napp, 1) * sigma
        x = np.vstack((x1, x2))
        y = np.vstack((np.ones((napp, 1)), np.ones((napp, 1)) * -1))

        xt1 = np.random.randn(ntest, 2) * sigma
        xt1[:, 0] *= 2
        xt2 = np.random.rand(ntest, 2) - 0.5

        xt2[:, 0] *= dilatation
        xt2[:, 1] += offset + vdil * xt2[:, 0] * xt2[:, 0]
        xt2[:, 1:2] += np.random.randn(ntest, 1) * sigma
        xtest = np.vstack((xt1, xt2))
        ytest = np.vstack((np.ones((ntest, 1)), np.ones((ntest, 1)) * -1))
        return x, y, xtest, ytest
    elif dataType == "checkers":
        dilatation = 4
        x = np.random.rand(2 * napp, 2) * dilatation - dilatation / 2
        y = np.ones((2 * napp, 1))
        for i in range(-dilatation / 2, dilatation / 2 + 1):
            for j in range(-dilatation / 2, dilatation / 2 + 1):
                if (i + j) % 2 == 0:
                    continue

                ind = np.where(
                    np.logical_and(
                        np.logical_and(
                            np.logical_and(x[:, 0] < i + 1, x[:, 0] >= i),
                            x[:, 1] < j + 1,
                        ),
                        x[:, 1] >= j,
                    )
                )
                y[ind, 0] = -1
        x += np.random.randn(2 * napp, 2) * sigma

        xtest = np.random.rand(2 * ntest, 2) * dilatation - dilatation / 2
        ytest = np.ones((2 * ntest, 1))
        for i in range(-dilatation / 2, dilatation / 2 + 1):
            for j in range(-dilatation / 2, dilatation / 2 + 1):
                if (i + j) % 2 == 0:
                    continue

                ind = np.where(
                    np.logical_and(
                        np.logical_and(
                            np.logical_and(xtest[:, 0] < i + 1, xtest[:, 0] >= i),
                            xtest[:, 1] < j + 1,
                        ),
                        xtest[:, 1] >= j,
                    )
                )
                ytest[ind, 0] = -1
        return x, y, xtest, ytest
    else:
        print("unsupported kind of data:", dataType)


############################################
############################################


def plotset(x, y, fig=None):
    indC1, buf = np.where(y == 1)
    indC2, buf = np.where(y == -1)
    # print indC2
    if fig != None:
        pl.figure(fig)  # recuperation d'une figure
    # else:
    #    pl.sci(fig)
    pl.plot(x[indC1, 0], x[indC1, 1], "r+")
    pl.plot(x[indC2, 0], x[indC2, 1], "b+")
    pl.show()
