# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:09:32 2013

@author: vguigue
"""

from frontiere import *
from dataset import *
from ml import *
import matplotlib.pyplot as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

pl.close('all')

x,y,xtest,ytest = dataset("gaussian", 100, 100, 1)

pl.figure(1)   # figure
plotset(x,y,1) # affichage des points

# decision : moindres carres ou perceptron
# w = perceptron(x,y,0.1,1000)
w = np.linalg.solve((x.T.dot(x)), (x.T.dot(y)))
print w

# maillage
xgrid = mesh(x)
# pl.figure(2) # affichage du maillage
# pl.plot(xgrid[:,0],xgrid[:,1],'+')

# evaluation du maillage
ygrid = xgrid.dot(w)
# fig = pl.figure(3) # affichage de l'évaluation du maillage
# ax = fig.gca(projection='3d')
# ax.scatter(xgrid[:,0],xgrid[:,1], ygrid)
# pl.savefig('grid.png',transparent=True)

# Affichage de la fonction de decision (3D)
fig = pl.figure(4)
decfunction(xgrid,ygrid)
# retracer les points pour mieux comprendre
figptr = pl.gcf();
ax = figptr.gca(projection='3d')
ax.scatter(x[np.where(y[:,0]==1),0],x[np.where(y[:,0]==1),1],0,s=10,c='r')
ax.scatter(x[np.where(y[:,0]==-1),0],x[np.where(y[:,0]==-1),1],0,s=10,c='b')
#pl.savefig('decmodel.png',transparent=True)

# Affichage de la frontiere de decision (2D) sur la figure 1
frontiere(xgrid,ygrid,1)
# pl.savefig('frontiere.png',transparent=True)

# Affichage des figures
pl.show()

