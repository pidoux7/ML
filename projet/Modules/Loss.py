import numpy as np


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass


class MSELoss(Loss):
    """
    Mean Square Error
    """

    def forward(self, y, yhat):
        """
        y : batch x d
        yhat :  batch x d
        Return coût: taille batch
        """
        return np.linalg.norm(y - yhat, axis=1) ** 2

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return -2 * (y - yhat)


class CElogSMLoss(Loss):
    """
    softmax + cross entropy en log
    """

    def forward(self, y, yhat):
        """
        y :  batch x d    ( ex pour multiclasse d = nb de classe avec un encodage one hot)
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return -np.sum(y * yhat, axis=1) + np.log(
            np.sum(np.exp(yhat), axis=1)
        )  # à vérifier

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return (
            np.exp(yhat) / np.sum(np.exp(yhat), axis=1).reshape((-1, 1)) - y
        )  # à vérifier
