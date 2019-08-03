import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling import Fittable1DModel, Parameter
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import find_lines_derivative

directory = os.getcwd() + '/data/'
files = os.listdir(directory)

list_fits = []
for i in files:
    if '.fit' in i:
        list_fits.append(i)
file_name = list_fits[0]  					  #need i
hdulist = fits.open(directory + file_name) 
data    = hdulist[0].data
header  = hdulist[0].header  
noise_mean = np.mean(data[0:][:])

left = 10**4
right = 0
SNR = 3

for i in range(len(data)):					  #finding the left/right edges of spectrum
    for j in range(len(data[:])):
        if data[i][j] > SNR * noise_mean:
            if j < left:
                left = j
            if j > right:
                right = j

lines = np.zeros((len(data), right-left+1))
for i in range(len(data)):
    q = 0
    for j in range(left, right):
        lines[i][q] = data[i][j]
        q+=1

slice = int(0.3 * (right - left))			  #weird koefficient
line = np.zeros(len(data))
for i in range(len(data)):
    line[i] = lines[i][slice]				  #result 1d line
not_real_lambda = range(len(line))

slice_light = left - 10 					  #slice of stray light -- continuum
line_light = np.zeros(len(data))
for i in range(len(data)):
    line_light[i] = data[i][slice]
line -= line_light

#finding emissions and absorptions lines
spectrum = Spectrum1D(flux=line*u.Jy, spectral_axis=not_real_lambda*u.um) #units only for work
lines1 = find_lines_derivative(spectrum, flux_threshold=100)

lam = []									  #wavelenghts of lines
max_flux = []								  #amplitudes of lines
num = len(lines1)
for i in range(num):
    max_flux.append(line[int(lines1['line_center'].value[i])])
    lam.append(lines1['line_center'].value[i])

std_hands = 1
compound_model = models.Gaussian1D(max_flux[0], lam[0], std_hands)
for i in range(1, num):
    compound_model += models.Gaussian1D(max_flux[i], lam[i], std_hands)
fitter = fitting.LevMarLSQFitter()
compound_fit = fitter(compound_model, not_real_lambda[0:1000], line[0:1000])

plt.figure(figsize=(8,5))                     #plot of spectrum
plt.plot(not_real_lambda, line, color='k')
plt.plot(not_real_lambda, compound_fit(not_real_lambda), color='darkorange')
plt.savefig(file_name + '_gauss_fit.png')
plt.close()

plt.figure(figsize=(8,5))                     #plot of errors
plt.plot(not_real_lambda, line - compound_fit(not_real_lambda), color='k')
plt.savefig(file_name + '_gauss_error.png')
plt.close()

amplitude = []
mean = []
stddev = []
FWHM = []
MIN_FLUX = 10000
for i in range(num):
    if compound_fit[i].amplitude.value > MIN_FLUX:
        amplitude.append(compound_fit[i].amplitude.value)
        mean.append(compound_fit[i].mean.value)
        stddev.append(compound_fit[i].stddev.value)
        FWHM.append(2.355 * compound_fit[i].stddev.value)

