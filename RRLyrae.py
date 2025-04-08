#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import os
import io

import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from astropy.table import Table
import numpy as np 

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy.timeseries import LombScargle

from joblib import Memory
cachedir = './joblib_cache'
memory = Memory(cachedir, verbose=0, bytes_limit=1e7)

import corner

@memory.cache
def get_gaia_query(q):
    #start = time.time()
    job = Gaia.launch_job_async(q)
    #print(f"Total time: {time.time()-start:0.2f} sec")
    return job.get_results()

rrlyrae = """SELECT TOP 100 *
        FROM gaiadr3.vari_rrlyrae
        WHERE pf IS NOT NULL 
        AND num_clean_epochs_g > 40
        ORDER BY source_id ASC
        """
rrlyrae_table = get_gaia_query(rrlyrae)
print(rrlyrae_table[0:9])

retrieval_type = 'EPOCH_PHOTOMETRY'          # Options are: 'EPOCH_PHOTOMETRY', 'MCMC_GSPPHOT', 'MCMC_MSC', 'XP_SAMPLED', 'XP_CONTINUOUS', 'RVS', 'ALL'
data_structure = 'INDIVIDUAL'   # Options are: 'INDIVIDUAL', 'COMBINED', 'RAW'
data_release   = 'Gaia DR3'     # Options are: 'Gaia DR3' (default), 'Gaia DR2'


datalink = Gaia.load_data(ids=rrlyrae_table['source_id'], data_release = data_release, retrieval_type=retrieval_type, \
                          data_structure = data_structure, verbose = False, output_file = None)
dl_keys  = [inp for inp in datalink.keys()]
dl_keys.sort()

print()
print(f'The following Datalink products have been downloaded:')
for dl_key in dl_keys:
    print(f' * {dl_key}')
    
dl_key      = dl_keys[0]  
product     = datalink[dl_key][0]
lightcurve  = product.to_table()    
sourceid_lc = lightcurve['source_id'][0]

for i in range(len(dl_keys)): 
    dl_key      = dl_keys[i]     # Try also with 'XP_CONTINUOUS_COMBINED.xml', 'MCMC_MSC_COMBINED.xml', 'MCMC_GSPPHOT_COMBINED.xml'
    product     = datalink[dl_key][0]
    product_tb  = product.to_table()                  # Export to Astropy Table object.
    source_ids  = list(set(product_tb['source_id']))  # Detect source_ids.
    print(f' There is data for the following Source IDs:')
    for source_id in source_ids:
        print(f'* {source_id}')


    inp_source = source_ids[0]                        # Replace "1" by "0" or "2" to show the data for the individual sources.
    product_tb = product_tb[product_tb['source_id'] == inp_source]

    print()
    print(f'Showing data for source_id {inp_source}')
    print(product_tb[0:5])                                   # Remove the '[0:5]' to display the entire table.
    
#need a list for each of the periods 
estimated_periods = []
average_gmags = []
#looping through each of the data sets to create periodograms and estimate the periods for each star
for i in range(len(dl_keys)): 
    dl_key      = dl_keys[i]     
    product     = datalink[dl_key][0]
    product_tb  = product.to_table()
    time = product_tb['time']
    flux = product_tb['mag']
    source_id = product_tb['source_id'][0]
    list_time = list(time)
    #creating periodogram
    frequency, power = LombScargle(list_time, flux, fit_mean=False, center_data=True).autopower(minimum_frequency=0.01,
                                         maximum_frequency=3, normalization='psd',
                                         samples_per_peak=30)
    #estimating the average gmag
    gmag = product_tb[np.where(product_tb['band'] == 'G')]['mag']
    average_gmag = 0
    for j in gmag: 
        average_gmag += 10**(j/-2.5)
    average_gmag = average_gmag
    average_gmag = -2.5 * np.log10(average_gmag)
    average_gmags.append(average_gmag)
    
    #plotting periodogram
    plt.figure()
    plt.plot(frequency, power) 
    plt.title(f'Lomb-Scargle Periodogram of RR Lyrae Star {source_id}')
    
    #finding what the period is 
    #need to make the power into a list and take out the units 
    #finding the maximum power and then finding the index to find where it is in  frequency/days
    power = list(power)
    index = power.index(max(power))
    period = frequency[index]
    estimated_periods.append(1/period)
    
    #writing the period and average gmag 
    txt = f'Period: {1/period}, Average G-mag: {average_gmag}'
    plt.figtext(0.5, -0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.xlabel('Frequency (1/Days)')
    plt.ylabel('Power')
    plt.show()
    

def beta(K, t, mag, w): 
    #creating X array
    X = np.ones((len(t),2*K+1))
    #the columns should be the length of t
    for i in range(len(t)):
        row = []
        #first row is A0 which is 1
        row.append(1.0)
        #the row should be the length of K+1
        for j in range(1, K+1):
            row.append(np.sin(w * t[i] * j))
            row.append(np.cos(w * t[i] * j))
        X[i] = row
    #print(X.shape)
    #print(len(mag))
    #getting the list of coefficients, Beta
    beta = np.linalg.lstsq(X, mag, rcond=None)[0]
    return beta, X

def ft(beta_matrix, X_matrix): 
    #getting flux by taking dot product of inverted X matrix and beta
    flux = np.dot(X_matrix,beta_matrix)
    #should return 1D array
    return flux 

def X(K, t, w): 
    Xmatrix = np.ones((len(t),2*K+1))
    for i in range(len(t)):
        row = []
        row.append(1.0)
        for j in range(1, K+1):
            row.append(np.sin(w * t[i] * j))
            row.append(np.cos(w * t[i] * j))
        Xmatrix[i] = row
    return Xmatrix
#creating training and test sets
from sklearn.model_selection import train_test_split
timetrain, timetest, magtrain, magtest  = \
train_test_split(star6time.value, star6mag.value, test_size = 0.2, train_size = 0.8)
K_list = []
training_validation = []
cross_validation = []
for K in range(1, 26): 
    K_list.append(K)
    #training 
    trainingbeta = beta(K, timetrain, magtrain, 2*np.pi/period6.value)
    training_betam = trainingbeta[0]
    training_Xm = trainingbeta[1]
    training_newft = ft(training_betam, training_Xm)
    training = sum((magtrain - training_newft)**2/training_newft)
    training_validation.append(training / len(timetrain))
    
    #cross validation 
    cross_X = X(K, timetest, 2*np.pi/period6.value)
    cross_newft = ft(training_betam, cross_X)
    cross = np.sum((magtest - cross_newft)**2 / (cross_newft))
    cross_validation.append(cross/len(timetest))
def mean_mag(flux): 
    avg_mag = 0 
    for i in range(len(flux)): 
        # mag = -2.5 * log(flux)
        calc_flux = 10**(flux[i]/-2.5)
        avg_mag += calc_flux
    avg_mag = avg_mag / len(flux)
    avg_mag = -2.5 * np.log10(avg_mag)
    return avg_mag
new_mean_mag = []
for i in range(len(dl_keys)): 
    
    dl_key      = dl_keys[i]     
    product     = datalink[dl_key][0]
    product_tb  = product.to_table()
    
    period = estimated_periods[i].value
    time = product_tb['time'].value
    mag = product_tb['mag'].value
    
    #gmag = product_tb['mag'][np.where(product_tb['band'] == 'G')]
    #time = product_tb['time'][np.where(product_tb['band'] == 'G')]
    
    timetrain, timetest, fluxtrain, fluxtest  = train_test_split(time, mag, \
                                                               test_size = 0.2, train_size = 0.8)
    
    
    model_beta = beta(13, timetrain, fluxtrain,  2*np.pi/period)[0]
    new_X = X(13, time,  2*np.pi/period)
    new_mag = ft(model_beta, new_X)
    #print(new_X.shape, new_mag.shape)
    mag = mean_mag(new_mag)
    new_mean_mag.append(mag)
residuals_avgmag = []
dl_sources = []
sorted_avg_gmag = []
for i in range(len(dl_keys)): 
    dl_key      = dl_keys[i]     
    product     = datalink[dl_key][0]
    product_tb  = product.to_table()
    
    #because the index is not the same for the Gaia table and the Data Link table, 
    #I have to find where they are the same index
    source_id = product_tb['source_id'][i]
    dl_sources.append(str(source_id))
    same_source = np.where(rrlyrae_table['source_id'] == source_id)
    sorted_avg_gmag.append(rrlyrae_table['int_average_g'][i])
    residual = rrlyrae_table['int_average_g'][same_source][0] - new_mean_mag[i]
    residuals_avgmag.append(residual)
def fourier_model(time, mag, period, K): 
    
    timetrain, timetest, fluxtrain, fluxtest  = train_test_split(time, mag, \
                                                               test_size = 0.2, train_size = 0.8)
    
    
    model_beta = beta(K, timetrain, fluxtrain,  2*np.pi/period)[0]
    new_X = X(K, time,  2*np.pi/period)
    new_mag = ft(model_beta, new_X)
    return new_mag
def lnGaussian(x): 
    mu = 1 
    sigma = 0.1
    Gauss = -((mu - x)**2 / (2 * sigma**2))
    return Gauss
def Gaussian(x): 
    mu = 1 
    sigma = 0.1
    Gauss = 1 / np.sqrt(2 * np.pi  * sigma**2) * np.exp(-(mu - x)**2 / (2 * sigma**2))
    return Gauss
samples = 1E+4
accept = 0

px_distribution = []
x = []
sigma = .1    

#getting x
x_in = 1
x.append(x_in)

#getting probability, p
p_x = lnGaussian(x_in)
px_distribution.append(p_x)

for i in range(1, int(samples)): 
    
    #getting x
    x_in = np.random.normal(loc = x[i-1], scale = sigma, size = 1)[0]
    x.append(x_in)

    #getting probability, p
    p_x = lnGaussian(x_in)
    px_distribution.append(p_x)
    
    #getting ratio 
    ratio = p_x - px_distribution[i-1]
    
    #choosing whether to accept or reject
    
    #accept
    if ratio >= 0: 
        accept += 1
        acceptance_fraction = accept / i
        print("Accepted", acceptance_fraction)
    
    #reject
    else: 
        #getting a random number to accept or reject, getting ratio to compare 
        random_no = np.random.uniform()
        
        #accepting
        if ratio >= random_no: 
            accept += 1
            acceptance_fraction = accept / i
            print("Accepted",acceptance_fraction)
        
        #rejecting
        else: 
            x[i] = x[i-1]
            acceptance_fraction = accept / i
            print("Rejected",acceptance_fraction)
        #rejecting
    ''' if random_no > ratio: 
            #override current x value with previous one
            x[i] = x[i-1]
            acceptance_fraction = accept / i
            print("Rejected",acceptance_fraction)
            
        #accepting  
        else: 
            accept += 1
            acceptance_fraction = accept / i
            print("Accepted",acceptance_fraction)'''
        
#return a sum of all of the array
def likelihood(predicted_mag, real, sigma, realsigma): 
    sigma = np.exp(sigma)
    totalerr = np.sqrt(realsigma**2 + sigma**2)
    return np.sum(-0.5 * ((real - predicted_mag)**2/totalerr**2 + np.log(2*np.pi*totalerr**2)))
 
def prior(slope, intercept, sigma): 
    lnp = 0 
    if (slope < -5) or (slope > 5) or (intercept > 6) or (intercept < -6) or (sigma < -2) or (sigma > 1):
        lnp = -np.inf
    return lnp

def Posterior(likelihood, prior): 
    if np.isfinite(prior): 
        posterior  = likelihood + prior
    else: 
        posterior = prior
    return posterior 

def predictedmag(slope, period, intercept): 
    return slope * period + intercept
    
#model MG = a × log [P/day] + b, where a and b are free parameters

#real data to pull from 
magnitude_list = np.array(magnitude18)
period_list = np.log10(periods18)
magnitude_err = np.array(magerrs18)

#magerrs18 = [] 
#magnitude18 = []
#periods18 = []

samples = 1E+4
accept = 0
step = 0.1

#slope
a = []
#intercept
b = []
#sigma
s = []

#initializing priors
slope, intercept, sigma = [-1,2, -1.5]
a.append(slope)
b.append(intercept)
s.append(sigma)
priors = prior(slope, intercept, sigma)

#initial predicted magnitude
mag = predictedmag(a[0], period_list, b[0])

#getting initial likelihood
likelihood_distribution = []
likelihood_distribution.append(likelihood(mag, magnitude_list, sigma, magnitude_err))

#getting initial posterior
posterior_distribution = [] 
posterior_distribution.append(Posterior(likelihood_distribution[0], priors))


#MCMC
for i in range(1, int(samples)): 

    #getting priors
    slope, intercept, sigma = np.random.normal([a[i-1], b[i-1], s[i-1]], scale = step)
    a.append(slope)
    b.append(intercept)
    s.append(sigma)
    current_prior = prior(slope, intercept, slope)

    #getting predicted magnitude
    mag = predictedmag(a[i], period_list, b[i])

    #getting the likelihood
    current_likelihood = likelihood(mag, magnitude_list, sigma, magnitude_err)
    likelihood_distribution.append(current_likelihood)

    #getting the posterior
    current_posterior = Posterior(current_likelihood, current_prior)
    posterior_distribution.append(current_posterior)

    #getting ratio for posterior
    ratio = current_posterior - posterior_distribution[i-1]

    #choosing whether to accept or reject

    #accept
    if ratio >= 0: 
        accept += 1
        acceptance_fraction = accept / i
        print("Accepted", acceptance_fraction)

    #reject
    else: 
        #getting a random number to accept or reject, using ratio to compare 
        random_no = np.random.uniform()

        #accepting
        if ratio >= random_no: 
            accept += 1
            acceptance_fraction = accept / i
            print("Accepted",acceptance_fraction)

        #rejecting
        else: 
            a[i] = a[i-1]
            b[i] = b[i-1]
            s[i] = s[i-1]
            acceptance_fraction = accept / i
            print("Rejected",acceptance_fraction)
#creating corner plot
posterior_distribution = np.array(posterior_distribution)
a = np.array(a)
b = np.array(b)
s = np.array(s)
data = np.vstack([a, b, s])
figure = corner.corner(data.T,
    labels=[
        r"a",
        r"b",
        r"$\sigma$"
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},)
figure.savefig('19icoroner.png')
#generating random values 
import random
randomrange = np.random.choice(a = 10000, size = 50, replace = True)
randoma = []
randomb = []
for i in randomrange: 
    randoma.append(a[randomrange])
    randomb.append(b[randomrange])
import pymc as pm
import arviz as az
from arviz import plot_trace as traceplot

# set up the model
with pm.Model() as model:
    
    # define priors
    miii = pm.Uniform("m", lower=-10, upper=10)
    biii = pm.Uniform("b", lower=-20, upper=20)
    logsigiii = pm.Uniform("logsig", lower=-10, upper=10)
    
    # define the log-likelihood function
    pm.Normal("obs", mu = miii*np.array(period_list)+biii, 
              sigma= np.sqrt(magnitude_err**2 + np.exp(logsigiii)**2), observed=np.array(magnitude_list))

    # now set up the model to run
    # default of PyMC is to use the no-turn sampler (NUTS)
    
    # pm.sample will run the sampler and store output in 'trace' 
    trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, discard_tuned_samples=True)
    
    # traceplot is a routine for plotting the 'traces' from the samples
    _ = traceplot(trace, var_names=["m", "b", "logsig"])
    # pm.summary provides some useful summary and convergance statistics
    az.summary(trace, var_names=["m", "b", "logsig"])
    _ = corner.corner(trace)
import pymc as pm
import arviz as az
from arviz import plot_trace as traceplot

# set up the model
with pm.Model() as model:
    
    # define priors
    miii = pm.Uniform("m", lower=-10, upper=10)
    biii = pm.Uniform("b", lower=-20, upper=20)
    logsigiii = pm.Uniform("logsig", lower=-10, upper=10)
    
    # define the log-likelihood function
    pm.Normal("obs", mu = miii*np.array(period_list)+biii, 
              sigma= np.sqrt(magnitude_err**2 + np.exp(logsigiii)**2), observed=np.array(magnitude_list))

    # now set up the model to run
    # default of PyMC is to use the no-turn sampler (NUTS)
    
    # pm.sample will run the sampler and store output in 'trace' 
    trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, discard_tuned_samples=True)
    
    # traceplot is a routine for plotting the 'traces' from the samples
    _ = traceplot(trace, var_names=["m", "b", "logsig"])
    # pm.summary provides some useful summary and convergance statistics
    az.summary(trace, var_names=["m", "b", "logsig"])
    _ = corner.corner(trace, tracequantiles=[0.16, 0.5, 0.84],
    show_titles=True)
fig = corner.corner(trace, quantiles=[.16, .50, .84], show_titles=True,
    title_kwargs={"fontsize": 12})

#creating corner plot
posterior_distribution = np.array(posterior_distribution)
a = np.array(a)
b = np.array(b)
s = np.array(s)
data = np.vstack([a, b, s])
figure = corner.corner(data.T, fig = fig, color = 'red', label = 'Pymc',
    labels=[
        r"m",
        r"b",
        r"logsig",
    ],
    #quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12})
plt.legend(['Model', 'Metropolis Hastings'])
figure.savefig('19iiicoroner.png')
import pymc as pm
import arviz as az
from arviz import plot_trace as traceplot

newperiod = np.array(newperiod)
newbprp = np.array(newbprp) 
newmagerr = np.array(newmagerr)


# set up the model
with pm.Model() as model:
    
    # define priors
    m = pm.Uniform("m", lower=-10, upper=10)
    b = pm.Uniform("b", lower=-20, upper=20)
    logsig = pm.Uniform("logsig", lower=-10, upper=10)
    
    # define the log-likelihood function
    pm.Normal("obs", mu = m*np.array(newperiod)+b, 
              sigma= np.sqrt(newmagerr**2 + np.exp(logsig)**2), observed=np.array(newbprp))

    # now set up the model to run
    # default of PyMC is to use the no-turn sampler (NUTS)
    
    # pm.sample will run the sampler and store output in 'trace' 
    trace = pm.sample(draws=1000, tune=1000, chains=2, cores=2, discard_tuned_samples=True)
    
    # traceplot is a routine for plotting the 'traces' from the samples
    _ = traceplot(trace, var_names=["m", "b", "logsig"])
    # pm.summary provides some useful summary and convergance statistics
    az.summary(trace, var_names=["m", "b", "logsig"])
    figure = corner.corner(trace)
color_excess = [] 
m_model = trace.posterior['m'].values.flatten()
b_model = trace.posterior['b'].values.flatten()
for i in range(len(newperiod)): 
    int_bprp = 0.11 * newperiod[i] + 0.86
    color_excess.append(newbprp[i] - int_bprp)
A_g = []
for i in range(len(color_excess)): 
    A_g.append(2 * color_excess[i])
import torch
ltensor = torch.FloatTensor(l)
btensor = torch.FloatTensor(bcoordinate)

fig = plt.figure(figsize = [15, 5])
plt.subplot(111, projection="aitoff")
plt.scatter(newl_tensor, newb_tensor, c = color_excess, alpha = 0.8)
plt.colorbar(label = '$E(G_{BP} - G_{RP})$')
plt.grid(True)
plt.title('Dust Map of Milky Way Galaxy', pad = 20)
plt.show()
fig.savefig('Q28.png')

