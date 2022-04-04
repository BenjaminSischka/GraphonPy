'''

Run the EM based algorithm -> iteration between the sampling (E-) and the estimation (M-) step.
@author: Benjamin Sischka

'''
import numpy as np
from GraphonPy.fitting import Estimator, TuneLambdaSplineRegAIC
from GraphonPy.sampling import Sample

def iterateEM(sortG,
              k, nKnots, canonical,
              n_steps, proposal, sigma_prop, use_origFct, averageType, use_stdVals,
              n_iter, rep_start, rep_end, it_rep_grow, rep_forPost,
              lambda_start, lambda_skip1, lambda_lim1, lambda_skip2, lambda_lim2, lambda_last_m,
              n_eval, trajMat=None,
              startWithEst=True, estGraphon=None, endWithSamp=True, raiseLabNb=False,
              returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
              makePlots=False, make_show=None, savefig=False, simulate=None, log_scale=False, dir_=None):
    result = type('', (), {})()
    lambda_ = lambda_start
    lambdas_ = np.array([])
    if trajMat is None:
        trajMat = np.zeros((0, n_eval, n_eval))
    if trajMat.shape[1:] != (n_eval,n_eval):
        raise TypeError('dimension of trajMat does not match n_eval')
    if returnLambList:
        result.lambdaList = []
    if returnSampList:
        result.sampleList = []
    if returnGraphonList:
        result.estGraphonList = []
    if not startWithEst:
        n_iter = n_iter+1
    # EM based algorithm
    for index in range(1,n_iter+1):
        labNb = index + raiseLabNb
        ### Update the graph
        if index > 1:
            print('Update graph')
            sample.updateGraph(use_stdVals =use_stdVals)
            if makePlots:
                sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dir_ + 'adjMat_' + (labNb-1).__str__() + '.png')
                sortG.showNet(make_show=make_show, savefig=savefig, file_=dir_ + 'network_' + (labNb-1).__str__() + '.png')
                if simulate:
                    sortG.showDiff(Us_type='est', EMstep_sign='(' + (labNb-1).__str__() + ')', make_show=make_show, savefig=savefig, file_=dir_ + 'Us_diffReal_' + (labNb-1).__str__() + '.png')
        ### Estimate Graphon
        if (index > 1) or startWithEst:
            print('Estimation')
            # determine the optimal penalization parameter
            estGraphonData=Estimator(sortG=sortG)
            if index in (list(range(lambda_skip1+1,lambda_lim1+1)) + list(range(lambda_skip2+1,lambda_lim2+1)) + list(range(n_iter-(lambda_last_m-1), n_iter+1))):
                lambda_ = TuneLambdaSplineRegAIC(estimator=estGraphonData, lambdaMin=0, lambdaMax=1000, paraDict={'k': k, 'nKnots': nKnots, 'canonical': canonical, 'Us_mult': None})
                if index in range(lambda_skip2+1,lambda_lim2+1):
                    lambdas_ = np.append(lambdas_, lambda_)
                if index == lambda_lim2:
                    lambda_ = lambdas_.mean()
            if returnLambList:
                result.lambdaList.append(lambda_)
            estGraphon=estGraphonData.GraphonEstBySpline(k=k, nKnots=nKnots, canonical=canonical, lambda_=lambda_, Us_mult=None, returnAIC=False)
            if makePlots:
                estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dir_ + 'graphon_est_' + labNb.__str__() + '.png')
        if returnGraphonList:
            result.estGraphonList.append(estGraphon)
        trajMat = np.append(trajMat, estGraphon.fct(np.arange(1, n_eval+1) / (n_eval+1), np.arange(1, n_eval+1) / (n_eval+1)).reshape(1, n_eval, n_eval), axis=0)
        ### Sample U's
        if (index < n_iter) or (endWithSamp and (rep_forPost > 0)):
            print('Sampling')
            rep = rep_start if (index < it_rep_grow) else (int(np.round((rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1)) * np.exp(np.log(rep_start / (rep_start**(n_iter-it_rep_grow) / rep_end)**(1/(n_iter-it_rep_grow-1))) * (index-it_rep_grow+1)))) if (index < n_iter) else (rep_forPost))
            sample=Sample(sortG=sortG,graphon=estGraphon,use_origFct=use_origFct)
            sample.gibbs(steps=n_steps,proposal=proposal,rep=rep,sigma_prop=sigma_prop,returnAllGibbs=returnAllGibbs,averageType=averageType,updateGraph=False,use_stdVals =None,printWarn=False)
            if makePlots:
                sample.showMove(useColor=True if simulate else False, EMstep_sign=labNb, make_show=make_show, savefig=savefig, file_=dir_ + 'Us_move_' + labNb.__str__() + '.png')
                if simulate:
                    sample.showMove(Us_type='real', useAllGibbs=True, EMstep_sign=labNb, make_show=make_show, savefig=savefig, file_=dir_ + 'UsMCMC_diffReal_' + labNb.__str__() + '.png')
            if returnSampList:
                result.sampleList.append(sample)
        eval("print('iteration completed:', index, ', penalizing parameter lambda:', lambda_, ',  number of Gibbs sampling stages:', rep)")
    result.sortG = sortG
    result.estGraphon = estGraphon
    result.sample = sample
    result.AIC = estGraphonData.AIC(lambda_ = lambda_, Us_mult=None)
    result.lambda_ = lambda_
    result.trajMat = trajMat
    return(result)

