import numpy as np
from tqdm import tqdm


class Sequential:
    """
    Ajouter des modules en automatisant les procédures de forward et backward.
    """

    def __init__(self, modules):  # module => list
        self._modules = modules

    def forward(self, input):
        """ "
        X matrice d'entrée : taille batch x input
        Return list
        """
        liste_forwards = [input]
        for i in range(0, len(self._modules)):
            liste_forwards.append(self._modules[i].forward(liste_forwards[-1]))
        return liste_forwards[1:]

    def update_parameters(self, gradient_step=1e-3):
        """
        Mise à jour des paramètres de chaque module selon le gradient avec un pas
        gradient_step : pas du gradient
        self._parameters : self.input x self.output
        """
        for module in self._modules:
            module.update_parameters(gradient_step=gradient_step)
            module.zero_grad()

    def backward_delta(self, liste_forwards, delta):  # +backward_update_gradient
        """
        Gradient du coût par rapport aux entrées en fonction de l’entrée input et des deltas de la couche suivante
        input: batch x self.input
        delta: batch x self.output
        Return delta de la couche actuelle : batch x self.input
        """
        liste_deltas = [delta]
        for i in range(len(self._modules) - 1, 0, -1):
            self._modules[i].backward_update_gradient(
                liste_forwards[i - 1], liste_deltas[-1]
            )
            liste_deltas.append(
                self._modules[i].backward_delta(liste_forwards[i - 1], liste_deltas[-1])
            )

        return liste_deltas


class Optim:
    """
    Classe permettant de condenser les
    """

    def __init__(self, net, loss, eps):
        self._net = net
        self._eps = eps
        self._loss = loss

    def step(self, batch_x, batch_y):
        """
        Réalise un forward, cout, backward avec un batch
        batch_y : batch de labels correspondants
        Return : le coût par rapport aux labels batch_y
        """
        # Forward
        liste_forwards = self._net.forward(batch_x)

        # Cout
        batch_y_hat = liste_forwards[-1]
        loss = self._loss.forward(batch_y, batch_y_hat)

        # backward
        delta = self._loss.backward(batch_y, batch_y_hat)
        list_deltas = self._net.backward_delta(liste_forwards, delta)

        # Mise à jour des paramètres
        self._net.update_parameters(self._eps)

        return loss

    def SGD(self, data_x, data_y, batch_size, epoch=100):
        """
        Effectue une descente de gradient.
        datax : jeu de données
        datay : labels correspondans
        batch_size : taille de batch
        iter : nombre d'itérations
        Return : loss
        """
        nb_data = len(data_x)
        nb_batches = nb_data // batch_size
        if nb_data % batch_size != 0:
            nb_batches += 1

        liste_loss = []

        for i in tqdm(range(epoch)):
            # permutter les données
            perm = np.random.permutation(nb_data)
            data_x = data_x[perm]
            data_y = data_y[perm]

            # découpe les l'array en liste d'arrays
            liste_batch_x = np.array_split(data_x, nb_batches)
            liste_batch_y = np.array_split(data_y, nb_batches)

            # Effectue la descente de gradient pour chaque batch
            loss_batch = 0
            for j in range(nb_batches):
                batch_x = liste_batch_x[j]
                batch_y = liste_batch_y[j]
                loss = self.step(batch_x, batch_y)
                loss_batch += loss.mean()
            loss_batch = loss_batch / nb_batches
            liste_loss.append(loss_batch)

        return liste_loss
