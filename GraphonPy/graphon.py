'''

Define a graphon class.
@author: Benjamin Sischka

'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
from copy import copy
from copy import deepcopy
from matplotlib.colors import LogNorm
import scipy.interpolate as interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

# auxiliary function to create the color bar for the graphon
def colorBar(mappable, ticks=None):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    if ticks is None:
        return(fig.colorbar(mappable, cax=cax))
    else:
        return(fig.colorbar(mappable, cax=cax, ticks=ticks))
#out: a color bar for heat plots

# auxiliary function to get a matrix as result of a discrete evaluation of the graphon function
# -> piecewise constant approximation of the graphon
def fctToMat(fct,size):
    # fct = specific graphon function, size = fineness of the return matrix
    if np.isscalar(size):
        us_i=us_j=np.linspace(0,1,size)
        size0 = size1 = size
    else:
        us_i=np.linspace(0,1,size[0])
        us_j=np.linspace(0,1,size[1])
        size0 = size[0]
        size1 = size[1]
    try:
        if fct(np.array([0.3,0.7]), np.array([0.3,0.7])).ndim == 1:
            if len(us_i) < len(us_j):
                mat=np.array([fct(us_i[i],us_j) for i in range(size0)])
            else:
                mat=np.array([fct(us_i,us_j[j]) for j in range(size1)])      
        else:
            mat=fct(us_i,us_j)
    except ValueError:
        warnings.warn('not appropriate graphon definition, slow from function to matrix derivation')
        print('UserWarning: not appropriate graphon definition, slow from function to matrix derivation')
        mat=np.zeros((size0, size1))
        for i in range(size0):
            for j in range(size1):
                mat[i,j]=fct(us_i[i],us_j[j])
    return(mat)
#out: squared matrix of the dimension (size,size) 

# auxiliary function to get a piecewise constant graphon function out of a matrix
def matToFct(mat):
    # mat = approx. graphon function on regular grid -> graphon matrix
    def auxFct(u,v):
        if np.isscalar(u):
            return(mat[np.minimum(np.floor(u*mat.shape[0]).astype(int), mat.shape[0]-1)][np.minimum(np.floor(v*mat.shape[1]).astype(int), mat.shape[1]-1)])
        else:
            return(mat[np.minimum(np.floor(u*mat.shape[0]).astype(int), mat.shape[0]-1)][:, np.minimum(np.floor(v*mat.shape[1]).astype(int), mat.shape[1]-1)])
    return(auxFct)
#out: piecewise constant bivariate function

# auxiliary function to get a vectorized version of a specified graphon function
def fctToFct(fct):
    # fct = specific graphon function
    try:
        if fct(np.array([0.3,0.7]), np.array([0.3,0.7])).shape != (2, 2):
            def auxFct(u,v):
                if np.isscalar(u) or np.isscalar(v):
                    return(fct(u,v))
                else:
                    if len(u) < len(v):
                        return(np.array([fct(u_i,v) for u_i in u]))
                    else:
                        return(np.array([fct(u,v_i) for v_i in v]).T)
            return(deepcopy(auxFct)) 
        else:
            return(deepcopy(fct)) 
    except ValueError:
        warnings.warn('function only accepts scalars')
        print('UserWarning: function only accepts scalars')
        def auxFct(u,v):
            if np.isscalar(u) and np.isscalar(v):
                return(fct(u,v))
            elif (not np.isscalar(u)) and np.isscalar(v):
                return(np.array([fct(u_i,v) for u_i in u]))
            elif np.isscalar(u) and (not np.isscalar(v)):
                return(np.array([fct(u,v_i) for v_i in v]))
            else:
                return(np.array([[fct(u_i,v_i) for v_i in v] for u_i in u]))
        return(deepcopy(auxFct)) 
#out: vectorized bivariate function


# Define Graphon Class
class Graphon:
    
    def __init__(self,fct=None,mat=None,size=501):
        # fct = specific graphon function, mat = approx. graphon function on regular grid, size = fineness of the graphon matrix
        if fct is None:
            if mat is None:
                raise TypeError('no informations about the graphon')
            self.mat = copy(np.asarray(mat))
            self.fct = matToFct(self.mat)
            self.byMat = True
        else:
            self.fct = fctToFct(fct)
            self.mat = fctToMat(fct,size)
            self.byMat = False
            if not mat is None:
                if not np.array_equal(np.round(fctToMat(fct,mat.shape), 5),np.round(mat, 5)):
                    warnings.warn('the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
                    print('UserWarning: the partitioning of the graphon in a grid \'mat\' is not exactly according to the graphon function \'fct\' or might be rotated')
    def showColored(self, vmin=None, vmax=None, vmin_=0.01, log_scale=False, ticks = [0, 0.25, 0.5, 0.75, 1], showColorBar=True, colorMap = 'plasma_r', fig_ax=None, make_show=True, savefig=False, file_=None):
        if (self.mat.min() < -1e-3) or (self.mat.max() > 1+1e-3):
            warnings.warn('graphon has bad values, correction has been applied -> codomain: [0,1]')
            print('UserWarning: graphon has bad values, correction has been applied -> codomain: [0,1]')
        self_mat = np.minimum(np.maximum(self.mat,0),1)
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if vmin is None:
            vmin = self_mat.min()
        vmin_diff = np.max([vmin_ - vmin, 0])
        if vmax is None:
            vmax = self_mat.max()
        plotGraphon = ax.matshow(self_mat + vmin_diff, cmap=plt.get_cmap(colorMap), interpolation='none', norm=LogNorm(vmin=vmin + vmin_diff, vmax=vmax + vmin_diff)) if log_scale else \
        ax.matshow(self_mat, cmap=plt.get_cmap(colorMap), interpolation='none', vmin=vmin, vmax=vmax)
        plt.xticks(self_mat.shape[1] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.yticks(self_mat.shape[0] * np.array(ticks) - 0.5, [(round(round(i,4)) if np.isclose(round(i,4), round(round(i,4))) else round(i,4)).__str__() for i in ticks])
        plt.tick_params(bottom=False)
        if showColorBar:
            ticks_CBar = [((10**(np.log10(vmin + vmin_diff) - i * (np.log10(vmin + vmin_diff) - np.log10(vmax + vmin_diff)) / 5)) if log_scale else ((i/5) * (vmax - vmin) + vmin)) for i in range(6)]
            cbar = colorBar(plotGraphon, ticks = ticks_CBar)
            cbar.ax.minorticks_off()
            cbar.ax.set_yticklabels(np.round(np.array(ticks_CBar) - (vmin_diff if log_scale else 0), 4))
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('plotGraphon' + (', cbar' if showColorBar else '')))
    def showExpDegree(self,size=101,norm=False,fmt='-',title=True,make_show=True,savefig=False,file_=None):
        if self.byMat:
            g_ = self.mat.mean(axis=0)
            us = np.linspace(0,1,self.mat.shape[1])
        else:
            g_ = fctToMat(fct=self.fct,size=(10*size,size)).mean(axis=0)
            us = np.linspace(0,1,size)
        if norm:
            plt.ylim((-1/20,21/20))
        plt.xlim((-1/20,21/20))
        plotDegree = plt.plot(us, g_, fmt)
        if title:
            plt.xlabel('u')
            plt.ylabel('g(u)')
        plt.gca().set_aspect(np.abs(np.diff(plt.gca().get_xlim())/np.diff(plt.gca().get_ylim()))[0])
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plotDegree)
#out: Graphon Object
#     fct = graphon function, mat = graphon matrix, byMat = logical whether graphon was specified by function or matrix
#     showColored = plot of the graphon function/matrix, showExpDegree = plot of the expected degree profile


# Define graphon generating function by predefined functions
def byExID(idX,size=101):
    # idX = id of function (see below), size = fineness of the graphon matrix
    examples = {
                1: lambda u,v: 1/2*(u+v),
                2: lambda u,v: ((1-u)*(1-v))**(1/1) * 0.8 + (u*v)**(1/1) * 0.85,
                3: lambda u,v: ((stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(u)) * stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(v))) + ((1 - stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(u))) * (1 - stats.norm(0,0.5).cdf(stats.norm(0,1).ppf(v))))) * 0.5,
               }
    return(Graphon(fct=examples[idX],size=size))
#out: graphon

# Define graphon by B-spline function
def byBSpline(tau, P_mat=None, theta=None, order=1, size=101):
    # tau = inner knot positions, P_mat/theta = parameters in form of matrix/vector, order = order of the B-splines
    if order == 0:
        if P_mat is None:
            if theta is None:
                raise ValueError('no information about the graphon values')
            nSpline1d = len(tau) -1
            P_mat = theta.reshape((nSpline1d, nSpline1d))
        else:
            if not theta is None:
                warnings.warn('parameter vector theta has not been used')
                print('UserWarning: parameter vector theta has not been used')
            theta = P_mat.reshape(np.prod(P_mat.shape))
        def grFct(x_eval, y_eval):
            vec_x = np.maximum(np.searchsorted(tau, np.array(x_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            vec_y = np.maximum(np.searchsorted(tau, np.array(y_eval, ndmin=1, copy=False)) -1, 0).astype(int)
            return(P_mat[vec_x][:,vec_y])
    else:
        if theta is None:
            if P_mat is None:
                raise ValueError('no information about the graphon values')
            theta = P_mat.reshape(np.prod(P_mat.shape))
        else:
            if not P_mat is None:
                warnings.warn('parameter matrix P_mat has not been used')
                print('UserWarning: parameter matrix P_mat has not been used')
            P_mat = theta.reshape((len(tau) -1, len(tau) -1))
        def grFct(x_eval, y_eval):
            x_eval_order = np.argsort(x_eval)
            y_eval_order = np.argsort(y_eval)
            fct_eval_order=interpolate.bisplev(x= np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(tau, tau, theta, order, order), dx=0, dy=0)
            return(eval('fct_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else '')))
    GraphonSpeci = Graphon(fct=grFct,size=size)
    GraphonSpeci.tau = tau
    GraphonSpeci.P_mat = P_mat
    GraphonSpeci.theta = theta
    GraphonSpeci.order = order
    return(GraphonSpeci)
#out: graphon

