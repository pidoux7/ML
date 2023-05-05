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
        return np.sum((y - yhat) ** 2, axis=1)

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return -2 * (y - yhat)


class CELoss(Loss):
    """
    cross entropy
    """

    def forward(self, y, yhat):
        """
        y : batch x d
        yhat :  batch x d
        Return coût: taille batch
        """
        return 1 - np.sum(yhat * y, axis=1)

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return yhat - y


class CElogLoss(Loss):
    """
    Cross entropy Log
    """

    def forward(self, y, yhat):
        """
        y : batch x d
        yhat :  batch x d
        Return coût: taille batch
        """
        return -(y * np.log(yhat + 1e-100) + (1 - y) * np.log(1 - yhat + 1e-100))

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return ((1 - y) / (1 - yhat + 1e-100)) - (y / yhat + 1e-100)


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
        return np.log(np.sum(np.exp(yhat), axis=1) + 1e-100) - np.sum(y * yhat, axis=1)

    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return (
            np.exp(yhat) / np.sum(np.exp(yhat), axis=1).reshape((-1, 1)) - y
        )  # à vérifier

class BCELoss(Loss):
    """
    Binary cross entropy
    """
    def forward(self, y, yhat):
        """
        y :  batch x d    ( ex pour multiclasse d = nb de classe avec un encodage one hot)
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return - (y*np.log(yhat + 1e-20) + (1-y)*np.log(1-yhat+ 1e-20))
    
    def backward(self, y, yhat):
        """
        y :  batch x d
        yhat : batch x d
        Return gradient de coût : batch x d
        """
        return ((1-y)/(1-yhat+ 1e-20)) - (y/yhat+ 1e-20)