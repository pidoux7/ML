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
    
class Conv1D(Module):
    """
    Module Conv1D
    """
    def __init__(self, k_size, chan_in, chan_out, stride):
        """
        K_size: taille de la fenêtre 
        chan_in: nb de canaux d'entrée
        chan_out: nb de canaux de sortie
        stride : pas de déplacement de la fenêtre 
        """
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        #std = 1. / np.sqrt(k_size * chan_in)
        #self._parameters = np.random.normal(0, std, (k_size, chan_in, chan_out))
        self._parameters = np.random.randn(k_size, chan_in, chan_out) * np.sqrt(2. / (k_size * chan_in))
        self._bias = np.zeros((1, chan_out))
        self._gradient = np.zeros((k_size, chan_in, chan_out))
        self._bias_gradient = np.zeros((1, chan_out))

    def forward(self, X):
        """
        X matrice d'entrée : taille batch x longueur x chan_in
        Return sorties du module : taille batch x longueur_prime x chan_out
        """
        X = X.reshape(X.shape[0],X.shape[1],-1)
        taille_img = np.size(X[0])//self._chan_in
        nb_img = np.size(X, axis=0)
        d_out = int((taille_img - self._k_size + 1) / self._stride)
        maxi = taille_img - self._k_size + 1
        output = np.zeros((nb_img, d_out, self._chan_out))

        for img in range(nb_img):
            for canal in range(self._chan_out):
                pas = -self._stride
                localisation = 0
                while pas + self._stride < maxi:
                    output[img, localisation, canal] = (
                        np.sum(self._parameters[:, :, canal] * X[img][pas + self._stride:pas + self._stride + self._k_size]) + self._bias[0, canal])
                    pas += self._stride
                    localisation += 1
        self._cache = (X, output)
        return output

    def backward_delta(self,X, delta):
        """
        Gradients du coût par rapport aux paramètres et biais en fonction de l’entrée du module input et des delta de la couche suivante
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: taille batch x longueur x chan_in
        delta: taille batch x longueur_prime x chan_out
        return gradient des parmetres : k_size x chan_in x chan_out
        + return gradient des biais : 1 x chan_out
        """
        _, output = self._cache
        X = X.reshape(X.shape[0],X.shape[1],-1)
        nb_img, d_out, chan_out = output.shape

        delta_in = np.zeros_like(X)
        for img in range(nb_img):
            for i in range(d_out):
                for j in range(chan_out):
                    # Mise à jour du gradient des paramètres
                    self._gradient[:, :, j] += delta[img, i, j] * X[img, i * self._stride:i * self._stride + self._k_size, :]

                    # Mise à jour du gradient du biais
                    self._bias_gradient[0, j] += delta[img, i, j]

                    # Mise à jour du gradient d'entrée
                    delta_in[img, i * self._stride:i * self._stride + self._k_size, :] += delta[img, i, j] * self._parameters[:, :, j]
                    
        
        self._gradient /= nb_img
        self._bias_gradient /= nb_img
        delta_in /= nb_img
        
        return delta_in

    def update_parameters(self, gradient_step=1e-3):
        """
        Mise à jour des paramètres du module selon le gradient avec un pas
        gradient_step : pas du gradient
        self._parameters : self.input x self.output
        """
        self._parameters -= gradient_step * self._gradient
        self._bias -= gradient_step * self._bias_gradient

    def zero_grad(self):
        """
        Réinitialiser le gradient à 0.
        """
        self._gradient = np.zeros_like(self._parameters)
        self._bias_gradient = np.zeros_like(self._bias)


class MaxPool1D(Module):
    def __init__(self, pool_size, stride):
        self._pool_size = pool_size
        self._stride = stride
        self._cache = None

    def forward(self, X):
        """
        X matrice d'entrée : taille batch x longueur x chan_in
        Return sorties du module : taille batch x longueur_prime x chan_in
        """
        X = X.reshape(X.shape[0], X.shape[1], -1)
        taille_img = np.size(X[0])//X.shape[2]
        nb_img = np.size(X, axis=0)
        d_out = int((taille_img - self._pool_size) / self._stride + 1)
        maxi = taille_img - self._pool_size + 1
        output = np.zeros((nb_img, d_out, X.shape[2]))

        for img in range(nb_img):
            for canal in range(X.shape[2]):
                pas = -self._stride
                localisation = 0
                while pas + self._stride < maxi:
                    window = X[img][pas + self._stride:pas + self._stride + self._pool_size, canal]
                    output[img, localisation, canal] = np.amax(window)
                    pas += self._stride
                    localisation += 1

        self._cache = (X, output)
        return output

    def backward_delta(self, X, delta):
        """
        Gradients du coût par rapport aux paramètres et biais en fonction de l’entrée du module input et des delta de la couche suivante
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: taille batch x longueur x chan_in
        delta: taille batch x longueur_prime x chan_in
        """
        _, output = self._cache
        X = X.reshape(X.shape[0], X.shape[1], -1)
        delta_in = np.zeros_like(X)
        nb_img, d_out, chan_out = output.shape
        

        for img in range(nb_img):
            for i in range(d_out):
                for j in range(chan_out):
                    window = X[img][i * self._stride:i * self._stride + self._pool_size, j]
                    max_idx = np.argmax(window)
                    max_coord = np.unravel_index(max_idx, window.shape)
                    delta_in[img, i * self._stride + max_coord[0], j] += delta[img, i, j]

        return delta_in

    def zero_grad(self):
        """
        Réinitialiser le gradient à 0.
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """
        Pas de paramètre à mettre à jour
        """
        pass


class flatten(Module):
    def __init__(self):
        self._input_shape = None

    def forward(self, X):
        self._input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward_delta(self, X, delta):
        return delta.reshape(self._input_shape)
    
    
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
