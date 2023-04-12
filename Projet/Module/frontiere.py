# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:50:23 2013

@author: vguigue
"""

import numpy as np
import matplotlib.pyplot as pl
from matplotlib import cm

###########################################################################

def mesh(x,n=30):
    """
    Creation d'un maillage à partir d'un ensemble de points 2D
    """
    mmin = x.min(0)   # recuperation du min de chaque colonne
    mmax = x.max(0)   # recuperation du max de chaque colonne
    xgrid1,xgrid2 = np.meshgrid(np.linspace(mmin[0],mmax[0],n), 
                        np.linspace(mmin[1],mmax[1],n) )
    return np.hstack((np.reshape(xgrid1,(n*n,1)), np.reshape(xgrid2,(n*n,1))))

###########################################################################

def frontiere(xgrid,ygrid,fig=None):
    """
    Affichage d'une frontiere de decision
    """
 
    n = np.sqrt(xgrid.shape[0])
    xgrid1 = xgrid[:,0:1].reshape((n,n));
    xgrid2 = xgrid[:,1:2].reshape((n,n));    
    ygridS = ygrid.reshape((n,n));  
    
    if fig != None:
        pl.figure(fig)        

    pl.contour(xgrid1,xgrid2,ygridS,[0])

###########################################################################
    
def decfunction(xgrid,ygrid,fig=None):
    """
    Affichage d'une fonction de decision
    """

    n = np.sqrt(xgrid.shape[0])
    xgrid1 = xgrid[:,0:1].reshape((n,n));
    xgrid2 = xgrid[:,1:2].reshape((n,n));    
    ygridS = ygrid.reshape((n,n));  
    
    figptr = pl.gcf();
    if fig != None:
        figptr=pl.figure(fig)        

    ax = figptr.gca(projection='3d')
    
    #ax.plot_surface(xgrid1,xgrid2, ygridS) # simple
    ax.plot_surface(xgrid1,xgrid2, ygridS, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    