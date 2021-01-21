'''

Define a graph class which includes the latent variables U_i, i=1,...,N and is used for the graphon estimation.
@author: Benjamin Sischka

'''
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from operator import itemgetter
import warnings
from copy import copy
from GraphonEst.graphon import fctToFct
from sklearn.metrics import euclidean_distances
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

# Define a simple Graph Class whith no U's included
class GraphClass:
    
    def __init__(self,A,labels=None):
        # A = adjacency matrix, labels = labels of the nodes
        if not (A.shape[0] == A.shape[1]):
            raise ValueError('adjacency matrix \'A\' is not quadratic')
        if not (np.max(np.abs(np.diag(A))) == 0):
            warnings.warn('adjacency matrix \'A\' contains loops')
        if A.dtype != int:
            if np.max(np.abs(A - A.astype(int))) != 0:
                warnings.warn('adjacency matrix \'A\' has been transformed into integer, some information has been lost')
            A=A.astype(int)
        self.N = A.shape[0]
        if labels is None:
            self.labels = {i: i for i in range(self.N)}
        else:
            if len(labels) != self.N:
                raise ValueError('length of labels does not coincide with the dimension of \'A\'')
            if len(np.unique(np.array(list(labels.values()) if (labels.__class__ == dict) else  labels))) != self.N:
                raise ValueError('labels are not unique')
            self.labels = {i: labels[i] for i in range(self.N)}
        if np.allclose(A, A.T, atol=1e-10):
            self.degree = {i: j for i, j in zip(list(self.labels.values()), np.sum(A, axis = 1))}
            self.symmetry = True
        else:
            warnings.warn('adjacency matrix \'A\' is not symmetric')
            self.inDegree = {i: j for i, j in zip(list(self.labels.values()), np.sum(A, axis = 0))}
            self.outDegree = {i: j for i, j in zip(list(self.labels.values()), np.sum(A, axis = 1))}
            self.symmetry = False
        self.A = copy(A)
#out: Graph Class
#     A = adjacency matrix, labels = labels of the nodes, N = order of the graph, (in-/out-)degree = dictionary of the (in-/out-)degrees,
#     symmetry = logical whether the adjacency matrix is symmetric


# Define an extended Graph Class with U's included
class ExtGraph(GraphClass):
    
    def __init__(self,A,labels=None,Us_real=None,Us_est=None,estByDegree=True):
        # A = adjacency matrix, labels = labels of the nodes, Us_real = real U's (in case of simulation), Us_est = estimated U's,
        # estByDegree = logical whether the node ordering should be estimated based on the degree
        GraphClass.__init__(self, A, labels)
        self.Ord_emp={i: j for (i, k), j in zip(sorted(self.degree.items(), key = itemgetter(1)), range(self.N))}
        self.Ord_emp={i: self.Ord_emp[i] for i in list(self.labels.values())}
        self.Us_emp={i: (np.linspace(0,1,self.N+2)[1:-1])[j] for (i, j) in self.Ord_emp.items()}
        self.Us_real, self.Ord_real = None, None
        if not Us_real is None:
            if len(Us_real) != self.N:
                raise ValueError('length of \'Us_real\' does not coincide with the dimension of \'A\'')
            self.Us_real={i: Us_real[i] for i in list(self.labels.values())} if Us_real.__class__ == dict else {list(self.labels.values())[i]: Us_real[i] for i in range(self.N)}
            self.Ord_real={i: j for (i, k), j in zip(sorted(self.Us_real.items(), key = itemgetter(1)), range(self.N))}
            self.Ord_real={i: self.Ord_real[i] for i in list(self.labels.values())}
        if not Us_est is None:
            if len(Us_est) != self.N:
                raise ValueError('length of \'Us_est\' does not coincide with the dimension of \'A\'')
            self.Us_est={i: Us_est[i] for i in list(self.labels.values())} if Us_est.__class__ == dict else {list(self.labels.values())[i]: Us_est[i] for i in range(self.N)}
            self.Ord_est={i: j for (i, k), j in zip(sorted(self.Us_est.items(), key = itemgetter(1)), range(self.N))}
            self.Ord_est={i: self.Ord_est[i] for i in list(self.labels.values())}
        else:
            if estByDegree:
                self.Us_est=copy(self.Us_emp)
                self.Ord_est=copy(self.Ord_emp)
            else:
                distMat = euclidean_distances(self.A)
                MatA = (-1/2) * distMat**2
                MatB = MatA - np.repeat(MatA.mean(axis=1), self.N).reshape(self.N, self.N) - np.tile(MatA.mean(axis=0), self.N).reshape(self.N, self.N) + np.repeat(MatA.mean(), self.N**2).reshape(self.N, self.N)
                eigVal, eigVec = np.linalg.eig(MatB)
                eigVec = eigVec / np.sqrt((eigVec**2).sum(axis=0))
                eigValSorting = np.flip(np.argsort(np.abs(eigVal)), axis=0)
                eigVal, eigVec = eigVal[eigValSorting], eigVec[:, eigValSorting]
                pos_ = np.argsort(np.argsort(eigVec[:, 0]))
                if [sum_[pos_ >= (self.N-1)/2].sum() < sum_[pos_ <= (self.N-1)/2].sum() for sum_ in [self.A.sum(axis=0)]][0]:
                    pos_ = pos_[::-1]
                self.Us_est = {list(self.labels.values())[i]: vals_[i] for vals_ in [np.linspace(0, 1, self.N + 2)[1:-1][pos_]] for i in range(self.N)}
                self.Ord_est={i: j for (i, k), j in zip(sorted(self.Us_est.items(), key = itemgetter(1)), range(self.N))}
                self.Ord_est={i: self.Ord_est[i] for i in list(self.labels.values())}
        self.sorting = None
        self.labels_=lambda: np.array(list(self.labels.values()))
        UsDict_=lambda selfObj = self, Us_type=None: None if (Us_type == None) else eval('selfObj.Us_' + Us_type)
        self.Us_=lambda Us_type=None: None if (UsDict_(Us_type=Us_type) is None) else np.array(list((UsDict_(Us_type=Us_type)).values()))
        self.degree_=lambda: np.array(list(self.degree.values()))
    def sort(self,Us_type='est'):  # Us_type='real','emp'
        if (Us_type == 'real') and (self.Us_real is None):
            warnings.warn('no real U\'s are given, sorting is done by est U\'s')
            Us_type='est'
        newOrd={{i: j for j, i in list(self.labels.items())}[k]: l for k, l in list(eval('self.Ord_' + Us_type + '.items()'))}
        newOrd_array = np.array([i for i, j in sorted(newOrd.items(), key = itemgetter(1))])
        self.A = self.A[newOrd_array][:, newOrd_array]
        self.labels={k: l for k, l in sorted({newOrd[i]: self.labels[i] for i in list(self.labels.keys())}.items())}
        self.Us_real=None if (self.Us_real is None) else {i: self.Us_real[i] for i in list(self.labels.values())}
        self.Us_est={i: self.Us_est[i] for i in list(self.labels.values())}
        self.Us_emp={i: self.Us_emp[i] for i in list(self.labels.values())}
        self.Ord_real=None if (self.Ord_real is None) else {i: self.Ord_real[i] for i in list(self.labels.values())}
        self.Ord_est={i: self.Ord_est[i] for i in list(self.labels.values())}
        self.Ord_emp={i: self.Ord_emp[i] for i in list(self.labels.values())}
        if self.symmetry:
            self.degree={i: self.degree[i] for i in list(self.labels.values())}
        else:
            self.inDegree={i: self.inDegree[i] for i in list(self.labels.values())}
            self.outDegree={i: self.outDegree[i] for i in list(self.labels.values())}
        self.sorting=Us_type
    def update(self, Us_real=None, Us_est=None):
        if not Us_real is None:
            if len(Us_real) != self.N:
                raise ValueError('length of \'Us_real\' does not coincide with the order of the graph')
            self.Us_real={i: Us_real[i] for i in list(self.labels.values())} if Us_real.__class__ == dict else {list(self.labels.values())[i]: Us_real[i] for i in range(self.N)}
            self.Ord_real={i: j for (i, k), j in zip(sorted(self.Us_real.items(), key = itemgetter(1)), range(self.N))}
            self.Ord_real={i: self.Ord_real[i] for i in list(self.labels.values())}
        if not Us_est is None:
            if len(Us_est) != self.N:
                raise ValueError('length of \'Us_est\' does not coincide with the order of the graph')
            self.Us_est={i: Us_est[i] for i in list(self.labels.values())} if Us_est.__class__ == dict else {list(self.labels.values())[i]: Us_est[i] for i in range(self.N)}
            self.Ord_est={i: j for (i, k), j in zip(sorted(self.Us_est.items(), key = itemgetter(1)), range(self.N))}
            self.Ord_est={i: self.Ord_est[i] for i in list(self.labels.values())}
        if ((not Us_real is None) and (self.sorting == 'real')) or ((not Us_est is None) and (self.sorting == 'est')):
            self.sort(Us_type=self.sorting)
    def makeCopy(self):
        copyObj = ExtGraph(A=copy(self.A),labels=copy(self.labels),Us_real=copy(self.Us_real),Us_est=copy(self.Us_est))
        if not self.sorting is None:
            copyObj.sort(self.sorting)
        return(copyObj)
    def showAdjMat(self, make_show=True, savefig=False, file_=None):
        plot1 = plt.imshow(self.A, cmap = 'Greys', interpolation = 'none', vmin = 0, vmax = 1)
        plt.locator_params(nbins=6)
        locs, labels = plt.xticks()
        x_lim, y_lim = plt.xlim(), plt.ylim()
        plt.xticks(locs, (locs+1).astype(int))
        plt.yticks(locs, (locs+1).astype(int))
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1)
    def showUsCDF(self, Us_type=None, make_show=True, savefig=False, file_=None):
        Us_type = Us_type if (not Us_type is None) else (self.sorting if (not self.sorting is None) else 'est')
        Us = self.Us_(Us_type)
        plot1 = plt.plot(np.concatenate(([0], np.repeat(np.sort(Us),2), [1])), np.repeat(np.arange(self.N+1)/self.N,2))
        plt.plot([0,1],[0,1])
        plt.xlim((-1/20,21/20))
        plt.ylim((-1/20,21/20))
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1)
    def showUsHist(self, Us_type=None, bins=20, alpha=0.3, showSplits=True, make_show=True, savefig=False, file_=None):
        Us_type = Us_type if (not Us_type is None) else (self.sorting if (not self.sorting is None) else 'est')
        if np.isscalar(bins):
            splitPos = np.linspace(0,1,bins+1)
        else:
            splitPos = bins
        if showSplits:
            plot1 = plt.plot(np.repeat(splitPos, 2), np.concatenate(([0], np.repeat(np.histogram(a=self.Us_(Us_type), bins=splitPos, density=True)[0], 2), [0])))
        hist1 = plt.hist(x=self.Us_(Us_type), bins=splitPos, density=True, alpha=alpha)
        plt.xlim((-1/20,21/20))
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval(('plot1, ' if showSplits else '') + 'hist1'))
    def showObsDegree(self, Us_type=None, absValues=False, norm=False, fmt='o', title = True, make_show=True, savefig=False, file_=None):
        Us_type = Us_type if (not Us_type is None) else (self.sorting if (not self.sorting is None) else 'est')
        Us = self.Us_(Us_type)
        if norm:
            plt.ylim((-(1/20)* self.N,(21/20)* self.N) if absValues else (-1/20,21/20))
        plt.xlim((1 - (1/20)* (self.N-1), self.N + (1/20)* (self.N-1)) if absValues else (-1/20,21/20))
        if title:
            plt.xlabel('$i$' if absValues else ('$u_i$' if (Us_type == 'real') else ('$\hat u_i^{\;' + Us_type + '}$')))
            plt.ylabel('$degree(i)$' if absValues else '$degree(i) \;/\; (N-1)$')
        plot1 = plt.plot((np.arange(self.N)+1) if absValues else Us[np.argsort(Us)], self.degree_()[np.argsort(Us)] * (1 if absValues else (1 / (self.N-1))), fmt)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1)
    def showExpDegree(self, graphon, Us_type=None, givenUs=False, absValues=False, norm=False, size=1000, fmt='o', title = True, make_show=True, savefig=False, file_=None):
        Us_type = Us_type if (not Us_type is None) else (self.sorting if (not self.sorting is None) else 'est')
        Us = self.Us_(Us_type)
        if norm:
            plt.ylim((-(1/20)* self.N,(21/20)* self.N) if absValues else (-1/20,21/20))
        plt.xlim((1 - (1/20)* (self.N-1), self.N + (1/20)* (self.N-1)) if absValues else (-1/20,21/20))
        if title:
            plt.xlabel('$i$' if absValues else ('$u_i$' if (Us_type == 'real') else ('$\hat u_i^{\;' + Us_type + '}$')))
            plt.ylabel('$g(u) \; \cdot \; (N-1) $' if absValues else ('$g(u_i)$' if (Us_type == 'real') else ('$g(\hat u_i^{\;' + Us_type + '})$')))
        Us_eval = Us if givenUs else np.linspace(0,1,size+2)[1:-1]
        d_ = np.array([np.mean(graphon.fct(Us[i], Us_eval)) for i in range(self.N)]) * ((self.N-1) if absValues else 1)
        plot1 = plt.plot((np.arange(self.N)+1) if absValues else Us[np.argsort(Us)], d_[np.argsort(Us)], fmt)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1)
    def showObsVsExpDegree(self, graphon, absValues=False, norm=False, size=1000, fmt1='C1o', fmt2 = 'C0--', title = True, make_show=True, savefig=False, file_=None):
        if self.Us_real is None:
            raise TypeError('no information about the real U\'s')
        if title:
            plt.xlabel('$E(degree(i)\,|\,u_i)$' if absValues else '$g(u_i)$')
            plt.ylabel('$degree(i)$' if absValues else '$degree(i) \;/\; (N-1)$')
        x_ = np.array([np.mean(graphon.fct(self.Us_('real')[i], np.linspace(0,1,size+2)[1:-1])) for i in range(self.N)]) * ((self.N-1) if absValues else 1)
        y_ = self.degree_() * (1 if absValues else (1 / (self.N-1)))
        if norm:
            lmts = [0, self.N] if absValues else [0,1]
        else:
            lmts = [np.max([np.min(x_),np.min(y_)]), np.min([np.max(x_),np.max(y_)])]
        plot1 = plt.plot(x_[np.argsort(x_)], y_[np.argsort(x_)], fmt1)
        plot2 = plt.plot(lmts, lmts, fmt2)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1, plot2)
    def showDiff(self, Us_type='est', fmt1 = 'C1o', fmt2 = 'C0--', EMstep_sign=None, title = True, make_show=True, savefig=False, file_=None):
        if self.Us_real is None:
            raise TypeError('no information about the real U\'s')
        plot1 = plt.plot(self.Us_('real')[np.argsort(self.Us_('real'))], self.Us_(Us_type)[np.argsort(self.Us_('real'))], fmt1)
        plot2 = plt.plot([0,1],[0,1], fmt2)
        if title:
            plt.xlabel('$u_i$')
            if EMstep_sign is None:
                EMstep_sign = Us_type
            plt.ylabel('$\hat u_i^{\;' + EMstep_sign + '}$')
        plt.xlim((-1/20,21/20))
        plt.ylim((-1/20,21/20))
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(plot1, plot2)
    def showNet(self, Us_type='est', splitPos=None, showColorBar=True, colorMap = 'jet', byDegree=False, with_labels=False, fig_ax=None, make_show=True, savefig=False, file_=None):
        # Us_type = type of U's using for coloring - if None -> no coloring
        if fig_ax is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        if (not Us_type is None) and showColorBar:
            hidePlot = ax.matshow(np.array([[0, 1]]), cmap=plt.get_cmap(colorMap), aspect='auto', origin='lower', extent=(-0.1, 0.1, -0.1, 0.1))
            hidePlot.set_visible(False)
        G_nx = nx.from_numpy_array(self.A)
        node_color = eval('cm.' + colorMap + '(' + ('self.Us_(Us_type)' if (splitPos is None) else 'np.searchsorted(splitPos, self.Us_(Us_type)) / len(splitPos)') + ')') if (not Us_type is None) else None
        net1 = nx.draw_networkx(G_nx, pos=nx.kamada_kawai_layout(G_nx), with_labels=with_labels, node_size=(self.degree_() / np.max(self.degree_()) *50) if byDegree else 35,
                                node_color=node_color, cmap=None, width=0.2, style='solid')
        plt.axis('off')
        if (not Us_type is None) and showColorBar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('bottom', size='4%', pad=0.2)
            cbar = fig.colorbar(hidePlot, orientation='horizontal', cax=cax)
        if make_show:
            plt.show()
        if savefig:
            plt.savefig(file_)
            plt.close(plt.gcf())
        else:
            return(eval('net1' + (', cbar' if ((not Us_type is None) and showColorBar) else '')))
    def logLik(self, graphon, Us_type=None, regardNoLoop=True, regardSym=False):
        Us_type = Us_type if (not Us_type is None) else (self.sorting if (not self.sorting is None) else 'est')
        Pi_mat = np.minimum(np.maximum(graphon.fct(self.Us_(Us_type), self.Us_(Us_type)), 1e-7), 1 - 1e-7)
        logProbMat = (self.A * np.log(Pi_mat)) + ((1 - self.A) * np.log(1 - Pi_mat))
        if regardNoLoop:
            np.fill_diagonal(logProbMat, 0)
        if regardSym:
            logProbMat[np.tril_indices(self.N,-1)] = 0
        return(np.sum(logProbMat))
#out: Extended Graph Object
#     A = adjacency matrix, labels = labels of the nodes, N = order of the graph, (in-/out-)degree = dictionary of the (in-/out-)degrees,
#     symmetry = logical whether the adjacency matrix is symmetric,
#     Us_real = dictionary of real U's, Us_est = dictionary of estimated U_i's, Us_emp = dictionary of empirical U's prespecified by Degree,
#     Ord_real = ordering of vertices by real U's, Ord_est = ordering of vertices by estimated U's, Ord_emp = empirical ordering by Degree,
#     sorting = type of applied ordering of the vertices, Us_() / labels_() / degree_() = transformation of dictionary in form of a vector
#     showAdjMat = graphical illustration of the adjacency matrix
#     sort = apply an ordering to the vertices
#     update = update components (real U's or estimated U's)
#     makeCopy = make a copy of the graph object
#     showAdjMat = plot the adjacency matrix
#     showUsCDF = show the cdf of the U's
#     showUsHist = show histogram of the U's
#     showObsDegree = show the profile of the observed degree
#     showExpDegree = show the profile of the expected degree given U's and graphon
#     showObsVsExpDegree = compare the profile of the observed vs. the expected degree
#     showDiff = show the difference between real and estimated U's
#     showNet = show graph as network
#     logLik = calculate log likelihood


# Define graph generating function given a specific graphon
def GraphByGraphon(graphon=None,w=None,Us=None,sizeU=None,randomSample=True,estByDegree=True,labels=None):
    # graphon = graphon, w = bivariate graphon function, Us = vector or dictionary of U's, sizeU = order of the graph (if 'Us' is not specified)
    # randomSample = logical whether U's should be random or equidistant within [0,1] (if 'Us' is not specified),
    # estByDegree = logical whether node ordering should be done based on degree, labels = labels of nodes
    if graphon is None:
        if w is None:
            raise TypeError('no informations about the graphon')
        w_fct = fctToFct(w)
    else:
        w_fct = graphon.fct
        if not w is None:
            x = np.linspace(0,1,101)
            if not np.array_equal(np.round(graphon.fct(x, x), 5), np.round(fctToFct(w)(x, x), 5)):  
                warnings.warn('function \'w\' is not according to the graphon')
    if Us is None:
        if sizeU is None:
            raise TypeError('no specification for the order of the graph')
        Us = np.random.uniform(0,1,sizeU) if randomSample else np.linspace(0,1,sizeU+2)[1:-1]
    else:
        Us = np.array([Us[lab_i] for lab_i in (list(Us.keys()) if (labels is None) else (list(labels.values()) if (labels.__class__ == dict) else labels))]) if (Us.__class__ == dict) else Us
        if not sizeU is None:
            if sizeU != len(Us):
                warnings.warn('parameter for size of U\'s has not been used')
    sizeU = len(Us)
    A = np.random.binomial(n=1, p=w_fct(Us,Us))
    A[np.tril_indices(sizeU)] = A.T[np.tril_indices(sizeU)]
    np.fill_diagonal(A, 0)
    return(ExtGraph(A=A,labels=labels,Us_real=Us,estByDegree=estByDegree))
#out: extended graph

