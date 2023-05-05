import numpy as np


class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        # Annule gradient
        pass

    def forward(self, input):
        # Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        # Calcule la mise a jour des parametres selon le gradient calculé et le pas de gradient_step
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        # Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        # Calcul la derivee de l'erreur
        pass


class TanH(Module):
    """
    Tangente hyperbolique
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        """
        Sorties du module pour les entrées passées en paramètre.
        input matrice des entrées: batch x input
        Return sorties du module : batch x output
        """
        return np.tanh(input)

    def update_parameters(self, gradient_step=0.001):
        """
        Pas de mise à jour des parametres
        """
        pass

    def backward_update_gradient(self, input, delta):
        """
        Pas de gradients des paramètres ni du biais
        """
        pass

    def backward_delta(self, input, delta):
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        return (1 - np.tanh(input) ** 2) * delta


class Sigmoide(Module):
    """
    Sigmoïde
    """

    def __init__(self):
        """
        Constructeur sans paramètres du module Sigmoïde.
        """
        super().__init__()

    def forward(self, input):
        """
        Sorties du module pour les entrées passées en paramètre.
        input matrice des entrées: batch x input
        Return sorties du module : batch x output
        """
        return 1 / (1 + np.exp(-input))

    def update_parameters(self, gradient_step=0.001):
        """
        Pas de mise à jour des parametres
        """
        pass

    def backward_update_gradient(self, input, delta):
        """
        Pas de gradients des paramètres ni du biais
        """
        pass

    def backward_delta(self, input, delta):
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        return (np.exp(-input) / (1 + np.exp(-input)) ** 2) * delta


class Softmax(Module):
    """
    Softmax
    """

    def __init__(self):
        """
        Constructeur sans paramètres du module softmax.
        """
        super().__init__()

    def forward(self, X):
        """
        Sorties du module pour les entrées passées en paramètre.
        input matrice des entrées: batch x input
        Return sorties du module : batch x output
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1).reshape((-1, 1))

    def update_parameters(self, gradient_step=0.001):
        """
        Pas de mise à jour des parametres
        """
        pass

    def backward_update_gradient(self, input, delta):
        """
        Pas de gradients des paramètres ni du biais
        """
        pass

    def backward_delta(self, input, delta):
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        
        
        return delta * (self.forward(input) * (1 - self.forward(input)))
    


class Relu(Module):
    """
    RELU
    """
    def __init__(self):
        """
        Constructeur sans paramètres du module Relu.
        """
        super().__init__()

    def forward(self, input):
        """
        Sorties du module pour les entrées passées en paramètre.
        input matrice des entrées: batch x input
        Return sorties du module : batch x output
        """
        return np.maximum(0, input)

    def backward_delta(self, input, delta):
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        relu_grad = (input > 0).astype(float)
        return delta * relu_grad
    
    def update_parameters(self, gradient_step=0.001):
        """
        Pas de mise à jour des parametres
        """
        pass

    def backward_update_gradient(self, input, delta):
        """
        Pas de gradients des paramètres ni du biais
        """
        pass
