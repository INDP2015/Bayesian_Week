# -*- coding: utf-8 -*-
"""
Created on Fri May  6 11:37:14 2016
    Simplified Bayesian model to make predictions about a bimodal
    version of the RSG task.
    The model relies on the ideas proposed in Jazerdy, Shadlen 2010
    
    It only implements the first two steps of the 3 stages,
    estimation and selection of interval, still need to 
    implement the production.
    
@author: soyunkope
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

pdf = sts.norm.pdf

# First create the stimulus
time = np.linspace(400, 1600, 1000)
unim = 800  # Unimodal mean
bim1 = 1000  # Narrow bimodal
bim2 = 1200  # Wide bimodal
sigm = 40  # Same sigma for all

# Stimuli distribution, collapsing time
stimUni = pdf(time, loc=unim, scale=sigm)
stimBm1 = stimUni + pdf(time, loc=bim1, scale=sigm)/2
stimBm2 = stimUni + pdf(time, loc=bim2, scale=sigm)/2

# Priors, either unimodal for both or bimodal
prUniA = pdf(time, loc=(unim+bim1)/2, scale=sigm*2)
prUniB = pdf(time, loc=(unim+bim2)/2, scale=sigm*4)
prBimA = stimBm1
prBimB = stimBm2
# Normalization of the priors
prUniA = prUniA/prUniA.sum()
prUniB = prUniB/prUniB.sum()
prBimA = prBimA/prBimA.sum()
prBimB = prBimB/prBimB.sum()

# Using the mean of Weber fractions that Jazerdy and Shadlen used
# Fro meassure and production
wm = np.mean([0.0935, 0.1028, 0.1436, 0.1208, 0.1053, 0.0481])
wp = np.mean([0.0858, 0.0635, 0.0894, 0.0583, 0.0623, 0.0625])


# Meassures & Production over the stimulus
trials = np.linspace(400, 1600, 33)
meassure = [pdf(time, loc=x, scale=x*wm) for x in trials]

esUnA = [meass*prUniA for meass in meassure]
esBiA = [meass*prBimA for meass in meassure]
esUnB = [meass*prUniB for meass in meassure]
esBiB = [meass*prBimB for meass in meassure]

eUA_CDF = [x.cumsum()/x.sum() for x in esUnA]
eBA_CDF = [x.cumsum()/x.sum() for x in esBiA]
eUB_CDF = [x.cumsum()/x.sum() for x in esUnB]
eBB_CDF = [x.cumsum()/x.sum() for x in esBiB]

BLS_MAP_UA = [[x.mean(), x.max()] for x in esUnA]
BLS_MAP_BA = [[x.mean(), x.max()] for x in esBiA]
BLS_MAP_UB = [[x.mean(), x.max()] for x in esUnB]
BLS_MAP_BB = [[x.mean(), x.max()] for x in esBiB]


exIn = int(time.shape[0]/4)
exTm = [time[exIn], time[exIn*2], time[exIn*3]]
mEx = [pdf(time, loc=exTm[x], scale=exTm[x]*wm) for x in range(len(exTm))]

# Plot the estimation
prSc = .8  # scale for the prior
legPD = ['Stimulus',
         'Scaled Unimodal Prior',
         'Scaled Bimodal Prior',
         'Example Meassures']
PDFs = [[stimBm1, prUniA*prSc, prBimA*prSc],
        [stimBm2, prUniB*prSc, prBimB*prSc]]

estim = [[[w[0] for w in BLS_MAP_UA],
          [x[1] for x in BLS_MAP_UA],
          [y[0] for y in BLS_MAP_BA],
          [z[1] for z in BLS_MAP_BA]],
         [[w[0] for w in BLS_MAP_UB],
          [x[1] for x in BLS_MAP_UB],
          [y[0] for y in BLS_MAP_BB],
          [z[1] for z in BLS_MAP_BB]]]

legES = ['BLS Unimodal',
         'MAP Unimodal',
         'BLS Bimodal',
         'MAP Bimodal']
f_PDF = plt.figure('PDFS')
for i in np.arange(2):
    axA = f_PDF.add_subplot(2, 2, i+1)
    axA.plot(time, PDFs[i][0], 'r',
             time, PDFs[i][1], 'g--',
             time, PDFs[i][2], 'b--')
    axA.hold(1)
    axA.plot(time, mEx[0], 'k:',
             time, mEx[i+1], 'k:', alpha=70)
    axA.set_ylim([0, 0.011])
    axB = f_PDF.add_subplot(2, 2, i+3)
    axB.plot(trials, estim[i][0]/np.sum(estim[i][0]), '*g-.',
             trials, estim[i][1]/np.sum(estim[i][1]), '+b--',
             trials, estim[i][2]/np.sum(estim[i][2]), '*g-.',
             trials, estim[i][3]/np.sum(estim[i][3]), '+b--',
             time, np.zeros(time.shape[0]))
    axB.set_ylim([0, 0.14])
    if i == 0:
        axA.set_title('Narrow Bimodal')
        axB.set_title('Bayesian Estimation')
        axA.set_xticks([])
    else:
        axA.set_title('Wide Bimodal')
        axA.set_xticks([])
        axA.set_yticks([])
        axB.set_yticks([])
        axA.legend(legPD)
        axB.legend(legES)
