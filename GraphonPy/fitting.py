'''

Fit the graphon for a given graph and known U's.
@author: Benjamin Sischka

'''
import numpy as np
import cvxopt
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import warnings
from copy import copy
from GraphonPy.graphon import Graphon


# Define an Estimator Class
class Estimator:

    def __init__(self, sortG):
        # sortG = sorted extended graph
        if sortG.sorting is None:
            warnings.warn('no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            print('UserWarning: no specification about Us_type (see sortG.sorting), empirical degree ordering is used')
            sortG.sort(Us_type='emp')
            warnings.warn('input graph is now sorted by empirical degree')
            print('UserWarning: input graph is now sorted by empirical degree')
        self.sortG = sortG
    def GraphonEstBySpline(self, k=1, nKnots=10, canonical=False, lambda_=50, Us_mult=None, returnAIC=False):
        # k = degree of splines (only 0 and 1 are implemented), nKnots = number of inner knots, canonical = logical whether to fit a canonical graphon,
        # lambda_ = parameter of penalty, Us_mult = multiple U-vectors in form of a matrix,
        # returnAIC = logical whether graphon estimate or AIC should be returned
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        m = Us_mult.shape[0]
        nSpline1d = nKnots + k - 1
        nSpline = nSpline1d ** 2
        t = np.linspace(- k / (nKnots - 1), 1 + k / (nKnots - 1), nKnots + 2 * k)
        if k == 0:
            freqVec = np.array([[np.sum(indexVec1 == i) for i in range(nSpline1d)] for indexVec1 in np.maximum(np.ceil(Us_mult * nSpline1d) - 1, 0)])
            freqVecCum = np.array([np.append([0], line1) for line1 in np.cumsum(freqVec, axis=1)])
            indexVecMat = np.array([np.vstack((freqVecCum_i[:-1], freqVecCum_i[1:])).T for freqVecCum_i in freqVecCum])
            def itemset(array, pos_i, pos_j, val):
                array[(pos_i[0]):(pos_i[1])][:, (pos_j[0]):(pos_j[1])] = val
                return (array)
            B = np.array([[itemset(array=np.zeros((self.sortG.N, self.sortG.N)), pos_i=indexVec[i], pos_j=indexVec[j], val=1) for i in range(nSpline1d) for j in range(nSpline1d)] for indexVec in indexVecMat])
            if canonical:
                A_part = (1 / nSpline1d) * np.repeat(1, nSpline1d)
        elif k == 1:
            B = np.array([np.array([interpolate.bisplev(x=np.sort(Us), y=np.sort(Us), tck=(t, t, np.lib.pad([1], (i, nSpline - i - 1), 'constant', constant_values=(0)), k, k), dx=0, dy=0) for i in np.arange(nSpline)])[:, np.argsort(np.argsort(Us)), :][:, :, np.argsort(np.argsort(Us))] for Us in Us_mult])
            if canonical:
                A_part = (1 / (nSpline1d - 1)) * np.concatenate(([1 / 2], np.repeat(1, nSpline1d - 2), [1 / 2]))
        else:
            raise TypeError('B-splines of degree k = ' + k.__str__() + ' have not been implemented yet')
        B_cbind = np.array([np.delete(B[l].reshape(nSpline, self.sortG.N ** 2), np.arange(self.sortG.N) * (self.sortG.N + 1), axis=1) for l in range(m)])
        if canonical:
            A1 = np.vstack((np.array([np.pad(np.append(-A_part, A_part), (nSpline1d * i, nSpline1d * (nSpline1d - i - 2)), 'constant', constant_values=(0, 0)) for i in range(nSpline1d - 1)]), np.identity(nSpline), -np.identity(nSpline)))
        else:
            A1 = np.vstack((np.identity(nSpline), -np.identity(nSpline)))
        A2 = np.array([]).reshape((nSpline, 0))
        for i in range(nSpline1d):
            for j in range(i + 1, nSpline1d):
                NullMat = np.zeros((nSpline1d, nSpline1d))
                NullMat[i, j], NullMat[j, i] = 1, -1
                A2 = np.hstack((A2, NullMat.reshape(nSpline, 1)))
        A2 = A2.T
        L_part = np.identity(nSpline1d)[:-1] - np.hstack((np.zeros((nSpline1d - 1, 1)), np.identity(nSpline1d - 1)))
        I_part = np.identity(nSpline1d)
        penalize = np.dot(np.kron(I_part, L_part).T, np.kron(I_part, L_part)) + np.dot(np.kron(L_part, I_part).T, np.kron(L_part, I_part))
        G_ = cvxopt.matrix(-A1)
        A_ = cvxopt.matrix(A2)
        theta_t = np.repeat(np.mean(self.sortG.degree_()) / self.sortG.N, nSpline)
        cvxopt.solvers.options['show_progress'] = False
        differ = 5
        index_marker = 1
        while (differ > 0.01 ** 2):
            Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
            mat1 = (B.swapaxes(0, 1) * ((self.sortG.A * (1 / Pi)) - ((1 - self.sortG.A) * (1 / (1 - Pi))))).swapaxes(0, 1)
            score = np.sum(np.sum(np.sum(mat1, axis=0), axis=1), axis=1) - np.sum(np.sum([np.diagonal(mat1[l], axis1=1, axis2=2).T for l in range(m)], axis=0), axis=0)
            mat2 = 1 / (Pi * (1 - Pi))
            fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(self.sortG.N ** 2, ), np.arange(self.sortG.N) * (self.sortG.N + 1)), B_cbind[l].T) for l in range(m)]), 0)
            P_ = cvxopt.matrix(fisher + lambda_ * penalize)
            q_ = cvxopt.matrix(-score + lambda_ * np.dot(theta_t, penalize))
            if canonical:
                h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline1d - 1 + nSpline, ), np.ones(nSpline, )))
            else:
                h_ = cvxopt.matrix(np.dot(A1, theta_t) + np.append(np.zeros(nSpline, ), np.ones(nSpline, )))
            b_ = cvxopt.matrix(np.dot(-A2, theta_t))
            delta_t = np.squeeze(np.array(cvxopt.solvers.qp(P=P_, q=q_, G=G_, h=h_, A=A_, b=b_)['x']))
            theta_tOld = copy(theta_t)
            theta_t = delta_t + theta_t
            differ = (1 / nSpline) * np.sum((theta_t - theta_tOld) ** 2)
            print('Iteration of estimating theta:', index_marker)
            index_marker = index_marker + 1
            if index_marker > 10:
                warnings.warn('Fisher scoring did not converge')
                print('UserWarning: Fisher scoring did not converge')
                print(theta_tOld)
                print(theta_t)
                print(np.round(theta_t - theta_tOld, 4))
                break
        if returnAIC:
            Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * theta_t, axis=3), 1e-5), 1 - 1e-5)
            mat2 = 1 / (Pi * (1 - Pi))
            fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(self.sortG.N ** 2, ), np.arange(self.sortG.N) * (self.sortG.N + 1)), B_cbind[l].T) for l in range(m)]), 0)
            df_lambda = np.trace(np.dot(np.linalg.inv(fisher + lambda_ * penalize), fisher))
            logProbMat = (self.sortG.A * np.log(Pi)) + ((1 - self.sortG.A) * np.log(1 - Pi))
            [np.fill_diagonal(logProbMat_i, 0) for logProbMat_i in logProbMat]
            return (-2 * np.sum(logProbMat) + 2 * df_lambda + ((2 * df_lambda * (df_lambda + 1)) / (((self.sortG.N ** 2 - self.sortG.N) * m) - df_lambda - 1)))
        else:
            if k == 0:
                def fct(x_eval, y_eval):
                    vec_x = np.maximum(np.ceil(np.array(x_eval, ndmin=1, copy=False) * nSpline1d) - 1, 0).astype(int)
                    vec_y = np.maximum(np.ceil(np.array(y_eval, ndmin=1, copy=False) * nSpline1d) - 1, 0).astype(int)
                    return (theta_t.reshape((nSpline1d, nSpline1d))[vec_x][:, vec_y])
            if k == 1:
                def fct(x_eval, y_eval):
                    x_eval_order = np.argsort(x_eval)
                    y_eval_order = np.argsort(y_eval)
                    fct_eval_order = interpolate.bisplev(x=np.array(x_eval, ndmin=1, copy=False)[x_eval_order], y=np.array(y_eval, ndmin=1, copy=False)[y_eval_order], tck=(t, t, theta_t, k, k), dx=0, dy=0)
                    return (eval('fct_eval_order' + (('[np.argsort(x_eval_order)]' + ('[:,' if len(y_eval_order) > 1 else '')) if len(x_eval_order) > 1 else ('[' if len(y_eval_order) > 1 else '')) + ('np.argsort(y_eval_order)]' if len(y_eval_order) > 1 else '')))
            self.graphonEst = Graphon(fct=fct)
            self.graphonEst.nKnots = nKnots
            self.graphonEst.t = t
            self.graphonEst.theta = theta_t
            self.graphonEst.order = k
            return (self.graphonEst)
    # additional AIC function to return AIC for a graphon estimate which has already been determined
    def AIC(self, lambda_ = 50, Us_mult=None):
        # lambda_ = parameter of penalty, Us_mult = multiple U-vectors in form of a matrix
        if not hasattr(self, 'graphonEst'):
            raise AttributeError('AIC can only be calculated after estimation (see self.GraphonEstBySpline())')
        if Us_mult is None:
            Us_mult = self.sortG.Us_(self.sortG.sorting).reshape(1, self.sortG.N)
        m = Us_mult.shape[0]
        nSpline1d = self.graphonEst.nKnots + self.graphonEst.order - 1
        nSpline = nSpline1d ** 2
        if self.graphonEst.order == 0:
            freqVec = np.array([[np.sum(indexVec1 == i) for i in range(nSpline1d)] for indexVec1 in np.maximum(np.ceil(Us_mult * nSpline1d) - 1, 0)])
            freqVecCum = np.array([np.append([0], line1) for line1 in np.cumsum(freqVec, axis=1)])
            indexVecMat = np.array([np.vstack((freqVecCum_i[:-1], freqVecCum_i[1:])).T for freqVecCum_i in freqVecCum])
            def itemset(array, pos_i, pos_j, val):
                array[(pos_i[0]):(pos_i[1])][:, (pos_j[0]):(pos_j[1])] = val
                return (array)
            B = np.array([[itemset(array=np.zeros((self.sortG.N, self.sortG.N)), pos_i=indexVec[i], pos_j=indexVec[j], val=1) for i in range(nSpline1d) for j in range(nSpline1d)] for indexVec in indexVecMat])
        if self.graphonEst.order == 1:
            B = np.array([np.array([interpolate.bisplev(x=np.sort(Us), y=np.sort(Us), tck=(self.graphonEst.t, self.graphonEst.t, np.lib.pad([1], (i, nSpline - i - 1), 'constant', constant_values=(0)), self.graphonEst.order, self.graphonEst.order), dx=0, dy=0) for i in np.arange(nSpline)])[:, np.argsort(np.argsort(Us)), :][:, :, np.argsort(np.argsort(Us))] for Us in Us_mult])
        B_cbind = np.array([np.delete(B[l].reshape(nSpline, self.sortG.N ** 2), np.arange(self.sortG.N) * (self.sortG.N + 1), axis=1) for l in range(m)])
        L_part = np.identity(nSpline1d)[:-1] - np.hstack((np.zeros((nSpline1d - 1, 1)), np.identity(nSpline1d - 1)))
        I_part = np.identity(nSpline1d)
        penalize = np.dot(np.kron(I_part, L_part).T, np.kron(I_part, L_part)) + np.dot(np.kron(L_part, I_part).T, np.kron(L_part, I_part))
        Pi = np.minimum(np.maximum(np.sum(B.swapaxes(1, 3) * self.graphonEst.theta, axis=3), 1e-5), 1 - 1e-5)
        mat2 = 1 / (Pi * (1 - Pi))
        fisher = np.sum(np.array([np.dot(B_cbind[l] * np.delete(mat2[l].reshape(self.sortG.N ** 2, ), np.arange(self.sortG.N) * (self.sortG.N + 1)), B_cbind[l].T) for l in range(m)]), 0)
        df_lambda = np.trace(np.dot(np.linalg.inv(fisher + lambda_ * penalize), fisher))
        logProbMat = (self.sortG.A * np.log(Pi)) + ((1 - self.sortG.A) * np.log(1 - Pi))
        [np.fill_diagonal(logProbMat_i, 0) for logProbMat_i in logProbMat]
        return (-2 * np.sum(logProbMat) + 2 * df_lambda + ((2 * df_lambda * (df_lambda + 1)) / (((self.sortG.N ** 2 - self.sortG.N) * m) - df_lambda - 1)))
# out: Estimator Object
#      GraphonEstBySpline = function for graphon estimation by B-splines
#         out: graphon plus parameters for B-spline function



# Define function to optimize the penalization parameter 'lambda_' based on the AIC
def TuneLambdaSplineRegAIC(estimator, lambdaMin=0, lambdaMax=1000, paraDict={}):
    # estimator = estimator object, [lambdaMin, lambdaMax] = range of potential lambda's
    # paraDict = dictionary of the parameters for the graphon estimation step
    for para_ in ['k', 'nKnots', 'canonical', 'Us_mult']:
        if not para_ in paraDict.keys():
            paraDict[para_] = {'k': 1, 'nKnots': 10, 'tau': None, 'canonical': False, 'Us_mult': None}[para_]
    def optFun(lambda_):
        return(estimator.GraphonEstBySpline(k=paraDict['k'], nKnots=paraDict['nKnots'], canonical=paraDict['canonical'],
                                            lambda_ = lambda_, Us_mult = paraDict['Us_mult'], returnAIC = True))
    return(optimize.fminbound(func = optFun, x1 = lambdaMin, x2 = lambdaMax, xtol=5e-01, maxfun = 50))
#output: optimal lambda

# Plot the trajectory of the sequence of w^(u,v) for exemplary positions u and v
def showTraject(trajMat,us_=None,make_show=True,savefig=False,file_=None):
    n_eval = trajMat.shape[1]
    if us_ is None:
        us_ = np.arange(1, n_eval+1) / (n_eval+1)
    plots1 = [plt.plot(trajMat[:,r,s], label = '$\hat w^{\;(m)}(' + us_[r].__str__() + ',\, ' + us_[s].__str__() + ')$') for r in range(n_eval) for s in range(n_eval) if s >= r]
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    legend1 = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('EM Iteration $m$')
    # plt.ylabel('$\hat w^{\;(m)}(u,v)$')
    if make_show:
        plt.show()
    if savefig:
        plt.savefig(file_)
        plt.close(plt.gcf())
    else:
        return(plots1, legend1)

