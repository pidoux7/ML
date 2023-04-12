import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        # Annule gradient
        pass

    def forward(self, X):
        # Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        # Calcule la mise a jour des parametres selon le gradient calculé et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass


class Linear(Module):
    """
    Module linéaire.
    """

    def __init__(self, input, output):
        """
        input : dimension de l'entrée
        output : nb de neurone dans la couche
        """
        self._input = input
        self._output = output
        self._parameters = 2 * (
            np.random.rand(self._input, self._output) - 0.5
        )  # initialisation aléatoire centrée en 0                    # peut etre ne pas mettre le 2 comme ca ca initialyse plus bas
        self._biais = np.random.random((1, self._output)) - 0.5
        self.zero_grad()  # initialise les gradients

    def forward(self, X):
        """ "
        X matrice d'entrée : taille batch x input
        Return sorties du module : taille batch x output
        """
        return np.dot(X, self._parameters) + self._biais

    def zero_grad(self):
        """
        Réinitialiser le gradient à 0.
        """
        self._gradient = np.zeros((self._input, self._output))
        self._biais_grad = np.zeros((1, self._output))

    def update_parameters(self, gradient_step=0.001):
        """
        Mise à jour des paramètres du module selon le gradient avec un pas
        gradient_step : pas du gradient
        self._parameters : self.input x self.output
        """
        self._parameters -= gradient_step * self._gradient
        self._biais -= gradient_step * self._biais_grad

    def backward_update_gradient(self, input, delta):
        """
        Gradients du coût par rapport aux paramètres et biais en fonction de l’entrée du module input et des delta de la couche suivante
        input: batch x self.input
        delta: batch X self.output
        return gradient des parmetres : self.input x self.output
        + return gradient des biais : 1 x self.output
        """
        self._gradient += np.dot(input.T, delta)
        self._biais_grad += np.sum(delta, axis=0)

    def backward_delta(self, input, delta):
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        return np.dot(delta, self._parameters.T)
