'''

Application of EM based graphon estimation.
@author: Benjamin Sischka

'''
import sys,os
## when running the file as script
dir1_ = os.path.dirname(__file__)
## when running the file in an interactive session
# dir1_ = os.path.dirname(os.path.realpath(''))
sys.path.append(dir1_)  # specify path to Module GraphonPy
from GraphonPy import *

import pickle

## specify graphic options
from matplotlib import use as use_backend
use_backend("Agg")  # "GTK3Agg"
make_show=False
plt.rcParams["axes.grid"] = False
plt.rcParams.update({'font.size': 14})
## specify figure sizes
figsize1 = (9, 4)
figsize2 = (9, 5)

## specify the considered object to analyze
simulate = False  # logical whether to used simulated or real data (-> knowledge about real U's)
idX = 2  # specify graphon to consider (only used if simulate == True)
data_ = 'alliances'  # specify data to consider (only used if simulate == False), alternatives: 'facebook', 'alliances', 'brain'
start_ = 0  # start value for global repetition of the algorithm
stop_ = 0  # stop value for global repetition of the algorithm

## graphic specifications
directory_ = os.path.join(dir1_, 'Graphics') + '/' + (('graphon_' + idX.__str__()) if simulate else data_) + '/'  # define directory to save graphics
if not os.path.exists(directory_):  # create directory for graphics, if it does not already exist
    if not os.path.exists(os.path.join(dir1_, 'Graphics')):
        os.mkdir(os.path.join(dir1_, 'Graphics'))
    os.mkdir(directory_)
savefig = True  # logical whether to save figures
add_nTry = True  # logical whether the identification number should be added to graphic file names (only used if savefig == True)
plotAll = True  # logical whether to plot auxiliary graphics too
log_scale = False if simulate else True  # logical whether to use log_scale for graphon plot

## about initialization
initialByDegree = True  # logical whether the degree is used for initial ordering or not (alternative: MDS)
randomInit = True  # logical whether to start with a random initialization of the U's (dominates 'initialByDegree')
useIndividGraphs = False  # logical whether to use the same or individual graphs for the global repetitions of the estimation routine (only used if simulate == True)
initRandomInit = False  # logical whether the unique graph used for each global repetition should initially have a random initialization
# (only used if randomInit == False and ((simulate == True and useIndividGraphs == False) or simulate == False))

initGraphonEst = False  # logical whether to make an initial estimate of the graphon
initCanonical = False  # logical whether to start with canoncial estimation (only used if initGraphonEst == True)
initPostDistr = False  # logical whether to calculate the initial posterior distribution (graphon specification necessary)
trueInit = False  # logical whether to start with true model (true ordering + true graphon, dominates 'randomInit', only used if simulate == True)
if initPostDistr and (not (initGraphonEst or trueInit)):
    raise TypeError('no initial graphon estimation available')

N = 100  # dimension of network (only used if simulate == True)
Us = None  # initial real U's (only used if simulate == True)
randomSample = False  # logical whether the real U's are a random or an equidistant sample (only used if simulate == True & Us == None)

## parameters for B-spline regression
k = 1  # order of B-splines (only 0 and 1 are implemented)
nKnots = 20  # number of inner knots for the B-spline basis
canonical = False  # logical whether a canonical representation should be fitted

## parameters for the sampling
n_steps = 150  # steps of Gibbs iterations
proposal='logit_norm'  # type of proposal to use for the sampling (alternatives: 'uniform')
sigma_prop = 2  # variance of sampling step (-> proposal distribution, only used if proposal == 'logit_norm')
use_origFct=True  # logical whether to use the graphon function itself or an discrete approx
averageType='mean'  # specify the kind of posterior average
use_stdVals=True  # logical whether to use standardized Us (-> equidistant)

## parameters for calculating the illustrated posterior distribution
rep_forPost = 25  # number of repetitions/sequences of the Gibbs sampling for calculating the posterior distribution
useAllGibbs = True  # logical whether to use all Gibbs repetitions for calculating the posterior or simply the mean
distrN_fine = 1000  # fineness of the posterior distribution -> number of evaluation points
ks_rel = np.array([0,0.25,0.75,1])  # relative k's for which the posterior is calculated
ks_abs = None  # absolute k's for which the posterior is calculated, if None they will be calculated below depending on ks_rel (dominates 'ks_rel')

## parameters for the EM algorithm
n_iter = 25  # number of EM interations
rep_start = 1  # start value for number of repetitions in the Gibbs sampling step
rep_end = 25  # end value for number of repetitions in the Gibbs sampling step
it_rep_grow = 5  # iteration from which rep starts to grow

lambda_start = 50  # start value for the penalization parameter
lambda_skip1 = 10  # iterations to skip before optimizing lambda
lambda_lim1 = (3) + lambda_skip1  # (.) = optimized lambdas not to use for the mean penalization parameter
lambda_skip2 = (2) + lambda_lim1  # (.) = iterations to skip before optimizing lambda
lambda_lim2 = (3) + lambda_skip2  # (.) = optimized lambdas to use for the mean penalization parameter
lambda_last_m = 3  # last m iterations at which lambda is optimized again
if np.any([lambda_lim2 >= (n_iter - lambda_last_m), lambda_skip1 <= it_rep_grow]):
    warnings.warn('specification of iterations for estimating lambda should be reconsidered')

## parameter for observing convergence
n_eval = 3  # number of evaluation points for the trajectory for observing convergence -> equidistant positions



### Define graphon (if simulation is considered -> simulate = True)

if simulate:
    graphon0=byExID(idX=idX, size=1000)
    graphonMin0, graphonMax0 = np.max([np.floor(np.min(graphon0.mat) / 0.05) *0.05, 0]), np.min([np.ceil(np.max(graphon0.mat) / 0.05) *0.05, 1])

result_list = []
for glob_ind in range(start_,stop_+1):

    seed_ = glob_ind
    np.random.seed(seed_)

    nTry = glob_ind.__str__() + '_'  # specify an identification for the run
    dirExt = directory_ + (nTry if add_nTry else '')


    if simulate:
        # plot true graphon
        graphon0.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_true.png')



    ### Define graph

    if simulate:
        if useIndividGraphs or (glob_ind == start_):
            graph0=GraphByGraphon(graphon=graphon0,Us=Us,sizeU=N,randomSample=randomSample,estByDegree=initialByDegree)
            graph0.sort(Us_type = 'est')
    else:
        if glob_ind == start_:
            graph0 = GraphFromData(data_=data_, dir_=dir1_, estByDegree=initialByDegree)
            N=graph0.N
            graph0.sort(Us_type = 'est')
    if (simulate and (useIndividGraphs or (glob_ind == start_))) or ((not simulate) and (glob_ind == start_)):
        print(np.sum(graph0.A) / N)  # average number of links per node
        print(np.sum(graph0.A) / ((N-1)*N))  # graph density
    if randomInit or ((initRandomInit and (glob_ind == start_)) and ((simulate and (not useIndividGraphs)) or (not simulate))):
        graph0.update(Us_est = np.random.permutation(np.linspace(0,1,N+2)[1:-1]))
    if (not randomInit) and ((simulate and (not useIndividGraphs)) or (not simulate)):
        if glob_ind == start_:
            Us_est_unique = copy(graph0.Us_est)
        else:
            graph0.update(Us_est = Us_est_unique)


    # plot adjacency matrix based on initial ordering
    graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_0.png')

    # plot network with initial ordering
    graph0.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_0.png')

    if simulate:
        # plot network with true ordering
        graph0.showNet(Us_type='real', make_show=make_show, savefig=savefig, file_=dirExt + 'network_true.png')

        # plot difference between initial estimates and real U's
        graph0.showDiff(Us_type='est', EMstep_sign='(0)', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_0.png')

        if plotAll:
            # define graph ordered by real U's
            graph0_trueSort=graph0.makeCopy()
            graph0_trueSort.sort(Us_type='real')

            # plot adjacency matrix based on true ordering
            graph0_trueSort.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_true.png')

            # plot network with true ordering
            graph0_trueSort.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_true.png')

            # plot observed vs expected degree profile
            graph0_trueSort.showObsDegree(absValues=False, norm=False, fmt = 'C1o', title = False, make_show=make_show, savefig=False)
            graphon0.showExpDegree(norm=False, fmt = 'C0--', title = False, make_show=make_show, savefig=False)
            plt.xlabel('(i) $u$   /   (ii) $u_i$')
            plt.ylabel('(i) $g(u)$   /   (ii) $degree(i) \;/\; (N-1)$')
            if make_show:
                plt.show()
            plt.savefig(dirExt + 'obsVSreal_expDegree.png')
            plt.close('all')



    ### Initial fit of graphon + initial posterior distribution of U_k

    if trueInit:
        if simulate:
            graph0.update(Us_est = graph0.Us_('real'))
        else:
            warnings.warn('real data example is considered, ground truth is unknown')

    if initGraphonEst:
        estGraphonData0 = Estimator(sortG=graph0)
        lambda_ = TuneLambdaSplineRegAIC(estimator=estGraphonData0, lambdaMin=0, lambdaMax=1000, paraDict={'k': k, 'nKnots': nKnots, 'canonical': canonical, 'Us_mult': None})
        estGraphon0 = estGraphonData0.GraphonEstBySpline(k=k, nKnots=nKnots, canonical=initCanonical, lambda_=lambda_, Us_mult=None, returnAIC=False)
        trajMat = estGraphon0.fct(np.arange(1, n_eval + 1) / (n_eval + 1), np.arange(1, n_eval + 1) / (n_eval + 1)).reshape(1, n_eval, n_eval)
        # plot initial graphon estimate
        estGraphon0.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_1.png')
        if simulate:
            # plot true vs initial estimated graphon
            graphonMin1, graphonMax1 = np.max([(np.min([graphonMin0, np.min(estGraphon0.mat)]) // 0.05) * 0.05, 0]), np.min([(np.max([graphonMax0, np.max(estGraphon0.mat)]) // 0.05 + 1) * 0.05, 1])
            plt.figure(1, figsize=figsize1)
            plt.subplot(121)
            graphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
            plt.subplot(122)
            estGraphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
            plt.tight_layout(w_pad=2, h_pad = 1.5)
            if make_show:
                plt.show()
            plt.savefig(dirExt + 'graphon_compare_1.png')
            plt.close('all')
    else:
        trajMat = None


    if trueInit:
        if simulate:
            estGraphon0 = graphon0
        else:
            warnings.warn('real data example is considered, ground truth is unknown')

    seed2_ = glob_ind + 1
    np.random.seed(seed2_)

    if ks_abs is None:
        ks_abs = np.unique(np.minimum(np.maximum(np.round(ks_rel * N).astype('int') - 1, 0), N - 1))  # absolute k's for which the posterior is calculated
    if initPostDistr:
        if rep_forPost == 0:
            warnings.warn('number of repetitions for calculating the posterior distribution is 0, no calculation is carried out')
        else:
            # apply Gibbs sampling to the initial U's given the graphon estimate based on the initial ordering
            sample0=Sample(sortG=graph0,graphon=estGraphon0,use_origFct=use_origFct)
            sample0.gibbs(steps=n_steps,proposal=proposal,rep=rep_forPost,sigma_prop=sigma_prop, returnAllGibbs=False, averageType=averageType, updateGraph=False, use_stdVals=None, printWarn=True)
            # calculate and plot the posterior distribution of U_k based on the initial graphon estimate, with k corresponding to the initial ordering
            sample0.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='(1)', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_0.png')
            sample0.updateGraph(use_stdVals=use_stdVals)
            # plot adjacency matrix based on initial ordering
            graph0.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_1.png')
            # plot network with initial ordering
            graph0.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_1.png')
            if simulate:
                # plot differences to real U's
                graph0.showDiff(Us_type='est', EMstep_sign='(1)', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_1.png')



    ### Sample U's and fit graphon again and again

    EM_obj = iterateEM(sortG=graph0,
                       k=k, nKnots=nKnots, canonical=canonical,
                       n_steps=n_steps, proposal=proposal, sigma_prop=sigma_prop, use_origFct=use_origFct, averageType=averageType, use_stdVals=use_stdVals,
                       n_iter=n_iter, rep_start=rep_start, rep_end=rep_end, it_rep_grow=it_rep_grow, rep_forPost=rep_forPost,
                       lambda_start=lambda_start, lambda_skip1=lambda_skip1, lambda_lim1=lambda_lim1, lambda_skip2=lambda_skip2, lambda_lim2=lambda_lim2, lambda_last_m=lambda_last_m,
                       n_eval=n_eval, trajMat=trajMat,
                       makePlots=plotAll, make_show=make_show, savefig=savefig, simulate=simulate, log_scale=log_scale, dir_=dirExt,
                       returnLambList=True, returnGraphonList=False, returnSampList=False, returnAllGibbs=False,
                       startWithEst=(not initGraphonEst) or (initGraphonEst and initPostDistr), estGraphon=estGraphon0 if initGraphonEst else None,
                       endWithSamp=True, raiseLabNb=initGraphonEst and initPostDistr)



    # plot adjacency matrix based on final ordering
    EM_obj.sortG.showAdjMat(make_show=make_show, savefig=savefig, file_=dirExt + 'adjMat_EM.png')

    # plot network with final ordering
    EM_obj.sortG.showNet(make_show=make_show, savefig=savefig, file_=dirExt + 'network_EM.png')

    # plot trajectory of graphon estimation sequence for specific positions u and v
    showTraject(trajMat=EM_obj.trajMat, make_show=make_show, savefig=savefig, file_=dirExt + 'trajectory_graphonSeq.png')

    # plot final graphon estimate
    EM_obj.estGraphon.showColored(log_scale=log_scale, make_show=make_show, savefig=savefig, file_=dirExt + 'graphon_est_EM.png')

    if simulate:
        # plot true vs estimated graphon
        graphonMin1, graphonMax1 = np.max([(np.min([graphonMin0, np.min(EM_obj.estGraphon.mat)]) // 0.05) * 0.05, 0]), np.min([(np.max([graphonMax0, np.max(EM_obj.estGraphon.mat)]) // 0.05 + 1) * 0.05, 1])
        plt.figure(1, figsize=figsize1)
        plt.subplot(121)
        graphon0.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
        plt.subplot(122)
        EM_obj.estGraphon.showColored(log_scale=log_scale, fig_ax=(plt.gcf(), plt.gca()), vmin=graphonMin1, vmax=graphonMax1, make_show=make_show, savefig=False)
        plt.tight_layout(w_pad=2, h_pad=1.5)
        if make_show:
            plt.show()
        plt.savefig(dirExt + 'graphon_compare_EM.png')
        plt.close('all')

        # plot difference between EM estimates and real U's
        EM_obj.sortG.showDiff(Us_type='est', EMstep_sign='EM', make_show=make_show, savefig=savefig, file_=dirExt + 'Us_diffReal_EM.png')

        if canonical:
            # plot true vs estimated marginalization
            graphon0.showExpDegree(size=1000,norm=False,fmt='-',title=False,make_show=make_show,savefig=False)
            EM_obj.estGraphon.showExpDegree(size=1000,norm=False,fmt='-',title=True,make_show=make_show,savefig=savefig,file_=dirExt + 'margin_compare_EM.png')

            # plot true vs estimated marginalization in direct comparison
            g1_ = fctToMat(fct=graphon0.fct, size=(10 * 100, 100)).mean(axis=0)
            g2_ = fctToMat(fct=EM_obj.estGraphon.fct, size=(10 * 100, 100)).mean(axis=0)
            minLim, maxLim = np.min(np.append(g1_, g2_)), np.max(np.append(g1_, g2_))
            lim_ = np.array([minLim, maxLim]) + (maxLim - minLim) * 1 / 20 * np.array([-1, 1])
            plt.xlim(lim_)
            plt.ylim(lim_)
            plt.plot(g1_, g2_, 'C1')
            plt.plot([minLim, maxLim], [minLim, maxLim], 'C0--')
            plt.xlabel('$g(u)$')
            plt.ylabel('$\hat g^{\;EM}(u)$')
            if make_show:
                plt.show()
            plt.savefig(dirExt + 'margin_compare2_EM.png')
            plt.close('all')

    if plotAll:
        # plot observed vs expected degree profile based on EM ordering
        EM_obj.sortG.showObsDegree(absValues=False, norm=False, fmt = 'C1o', title=False, make_show=make_show, savefig=False)
        EM_obj.estGraphon.showExpDegree(norm=False, fmt = 'C0--', title=None, make_show=make_show, savefig=False)
        plt.xlabel('(i) $u$   /   (ii) $\hat u_i^{\;EM}$')
        plt.ylabel('(i) $\hat g^{\;EM}(u)$   /   (ii) $degree(i) \;/\; (N-1)$')
        if make_show:
            plt.show()
        plt.savefig(dirExt + 'obsVsEM_expDegree.png')
        plt.close('all')

    if rep_forPost != 0:
        # calculate and plot the posterior distribution of U_k based on the final graphon estimate, with k corresponding to the final ordering
        EM_obj.sample.showPostDistr(ks_lab=None, ks_abs=ks_abs, Us_type_label='est', Us_type_add='est', distrN_fine=distrN_fine, useAllGibbs=True, EMstep_sign='EM', figsize=figsize2, mn_=None, useTightLayout=True, w_pad=2, h_pad = 1.5, make_show=make_show, savefig=savefig, file_=dirExt + 'postDistr_EM.png')


    result_ = np.array([nTry, EM_obj.sortG.logLik(graphon = EM_obj.estGraphon), EM_obj.AIC, EM_obj.lambda_])
    print(result_)
    result_list.append(result_)

    graph_simple = {'A': EM_obj.sortG.A, 'labels': EM_obj.sortG.labels_(),
                    'Us_real': EM_obj.sortG.Us_('real'), 'Us_est': EM_obj.sortG.Us_('est')}
    graphon_simple = {'mat': EM_obj.estGraphon.mat, 'nKnots': EM_obj.estGraphon.nKnots,
                      't': EM_obj.estGraphon.t, 'theta': EM_obj.estGraphon.theta,
                      'order': EM_obj.estGraphon.order}
    if rep_forPost != 0:
        sample_simple = {'U_MCMC': EM_obj.sample.U_MCMC, 'U_MCMC_std': EM_obj.sample.U_MCMC_std,
                         'U_MCMC_all': EM_obj.sample.U_MCMC_all, 'accepRate': EM_obj.sample.accepRate,
                         'Us_new': EM_obj.sample.Us_new, 'Us_new_std': EM_obj.sample.Us_new_std}
    else:
        sample_simple = None

    with open(dirExt + 'final_result.pkl', 'wb') as output:
        pickle.dump(result_, output, protocol=3)
        pickle.dump(graph_simple, output, protocol=3)
        pickle.dump(graphon_simple, output, protocol=3)
        pickle.dump(sample_simple, output, protocol=3)


    # add parameter settings to a csv file
    fname = directory_ + '_register.csv'
    if not os.path.isfile(fname):
        with open(fname, 'a') as fd:
            fd.write('nTry; logLik; AIC; seed; seed2; initialByDegree; randomInit; initGraphonEst; initCanonical; initPostDistr; trueInit; N; k; nKnots; canonical; n_steps; sigma_prop; averageType; use_stdVals; rep_forPost; n_iter; rep_start; rep_end; it_rep_grow; lambda_start; lambda_skip1; lambda_lim1; lambda_skip2; lambda_lim2; lambda_last_m; lambda_; \n')
    with open(fname, 'a') as fd:
        fd.write(nTry + ';' + result_[1] + ';' + result_[2] + ';' + seed_.__str__() + ';' + seed2_.__str__() + '; ' + initialByDegree.__str__() + '; ' + randomInit.__str__() + '; ' + initGraphonEst.__str__() + '; ' + initCanonical.__str__() + '; ' + initPostDistr.__str__() + '; ' + trueInit.__str__() + '; ' + N.__str__() + '; ' + k.__str__() + '; ' + nKnots.__str__() + '; ' + canonical.__str__() + '; ' + n_steps.__str__() + '; ' + sigma_prop.__str__() + '; ' + averageType + '; ' + use_stdVals.__str__() + '; ' +
                 rep_forPost.__str__() + '; ' + n_iter.__str__() + '; ' + rep_start.__str__() + '; ' + rep_end.__str__() + '; ' + it_rep_grow.__str__() + '; ' + lambda_start.__str__() + '; ' + lambda_skip1.__str__() + '; ' + lambda_lim1.__str__() + '; ' + lambda_skip2.__str__() + '; ' + lambda_lim2.__str__() + '; ' + lambda_last_m.__str__() + '; ' + np.round(EM_obj.lambda_, 4).__str__() + '; \n')


    print('\n\n\nGlobal repetition complete:    ' + glob_ind.__str__() + '\n\n\n\n\n\n\n')


[print(entry_i) for entry_i in result_list]
print('label of best repetition:', [entry_i[0] for entry_i in result_list][np.argmin([float(entry_i[2]) for entry_i in result_list])])

