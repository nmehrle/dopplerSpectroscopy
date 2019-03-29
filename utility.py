import pickle
import numpy as np
import os

from scipy import ndimage as ndi
from scipy import constants, optimize, interpolate

from astropy.io import fits
from astropy.time import Time

import barycorrpy
from barycorrpy.utils import get_stellar_data


#-- File I/O
def readFile(dataFile):
  '''
    Reads data from file. Currently supports fits, pickle

    Parameters:
      dataFile (str) : Path to datafile to read

    Returns:
      data () : Data found in file, could be multiple types
  '''
  extension = dataFile.split('.')[-1]
  if extension == 'fits':
    data = fits.getdata(dataFile)
  elif extension == 'pickle':
    try:
      with open(dataFile, 'rb') as f:
        data = pickle.load(f)
    except UnicodeDecodeError:
      with open(dataFile, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
  else:
    raise ValueError('Error reading "'+dataFile+'" Currently only fits and pickle are supported')

  return data

def makePaths(path):
  '''
    Given a path, this function goes through all included subpaths and creates them if they do not already exist

    Parameters:
      path (str): directory path e.g. "this/is/a/path/"
  '''
  dirList = path.split('/')

  for i in range(1, len(dirList)):
    subPath = '/'.join(dirList[:i])
    if os.path.exists(subPath):
      pass
    else:
      os.mkdir(subPath)
###

#-- Object Manipulation
def dictDepth(d, level=0, empty=1):
  '''
    Gives the depth of a dictionary

    e.g.
    {}     -> (empty)
    {a: 1} -> 1
    {a: {b: 2}} -> 2

    Parameters:
      d (dict) : the dictionary to find the depth of
      level (int) : keeps track of depth recursively, change to modify the depth recorded
      empty (int) : (optional) the depth of the empty dict: {}

    Returns:
      level (int): the depth level of the dictionary
  '''
  if d == {}:
    return level+empty
  elif not isinstance(d, dict) or not d:
    return level
  return max(dictDepth(d[k], level + 1) for k in d)
###

#-- Math
def snr(data):
  ''' 
    Gives the SNR of a data, recommended to be a vector

    Parameters: 
      data (array) : the input array

    Returns:
      snr (float) : the snr of the data
  '''
  return np.mean(data)/np.std(data)

def getLocalMinima(data, neighborhood_size=20):
  '''
    Finds all points that are the minimum in a neighborhood of (neighborhood_size) about themselves

    Parameters:
      data (array): input data vector
      neighborhood_size (int) : the size of the neighborhood to define around each point

    Returns:
      localMinima (list) : list of indicies of local minima in data
  '''
  minima = ndi.minimum_filter(data, neighborhood_size)
  is_minima = (data == minima)
  return np.where(is_minima)[0]

def getLocalMaxima(data, neighborhood_size=20):
  '''
    Finds all points that are the maximum in a neighborhood of (neighborhood_size) about themselves

    Parameters:
      data (array): input data vector
      neighborhood_size (int) : the size of the neighborhood to define around each point

    Returns:
      localMaxima (list) : list of indicies of local maxima in data
  '''
  maxima = ndi.maximum_filter(data, neighborhood_size)
  is_maxima = (data == maxima)
  return np.where(is_maxima)[0]

def rfft(a, pad=True, axis=-1, returnPadLen = False):
  '''
    Wrapper for np.fft.rfft which automatically pads to length base-2

    Parameters:
      a (array): data to take fft of, fft is taken along axis (axis)
      pad (bool) : whether or not to pad to len base-2
      axis (int) : which axis to take fft along in a
      returnPadLen (bool) : if True, returns the length to which data has been padded

    Returns:
      fft (array): the fft of data along axis
      padLen (int): the length to which input data was padded
  '''
  n = np.shape(a)[axis]
  power = int(np.ceil(np.log2(n)))
  if pad:
    if returnPadLen:
      return np.fft.rfft(a, 2**power, axis=axis),2**power
    return np.fft.rfft(a, 2**power, axis=axis)
  else:
    return np.fft.rfft(a, axis=axis)

def correlate(target, reference, fourier_domain = False):
  """ Correlation function with option to pass data already in the 
    Fourier domain.
  """
  if not fourier_domain:
    n = len(reference)
    target = target - np.mean(target,1, keepdims=True)
    target = rfft(target)

    reference = reference - np.mean(reference)
    reference = rfft(reference)

  fft_corr = np.conj(reference) * target
  
  if not fourier_domain:
    return ifftCorrelation(fft_corr, n)

  return fft_corr
  
def ifftCorrelation(fft_corr, n=None):
  """ Inverts the correlation matrix from correlate, 
    Applies the transformation to correct for circular/non-circular ffts
  """
  corr = np.fft.irfft(fft_corr)  
  if n == None:
    m = np.shape(corr)[1]
    mid_point = int(m/2)
    second_half = corr[...,:mid_point]
    first_half  = corr[...,mid_point:]

    corr = np.concatenate((first_half , second_half), axis = -1)
    return corr
  else:
    m = int(n/2)
    return np.concatenate((corr[...,-m:] , corr[...,:m]), axis = -1)

def shiftData(y, shift, error=None, ext=3):
  """ Shifts data considering errors 
  """
  x = np.arange(len(y))

  weights = None
  if error is not None:
    weights = 1/error

  ip = interpolate.splrep(x, y, weights)
  interpolated = interpolate.splev(x - shift, ip, ext=ext)

  return interpolated

def fourierShift1D(y, shift, n=-1, fourier_domain=False):
  """ Shifts data quickly,
    Option to pass data already in fourier domain
  """
  if not fourier_domain:
    m = len(y)
    y, n = rfft(y,returnPadLen=True)

  fft_shift = ndi.fourier_shift(y, shift, n)

  if not fourier_domain:
    return np.fft.irfft(fft_shift)[:m]

  return fft_shift

def fourierShift2D(a, shifts, n=-1, fourier_domain=False):
  if not fourier_domain:
    m = np.shape(a)[1]
    a,n = rfft(a, returnPadLen=True)

  temp = []
  for i in range(len(a)):
    temp.append(fourierShift1D(a[i], shifts[i], n=n, fourier_domain=True))

  if not fourier_domain:
    return np.fft.irfft(temp)[:,:m]

  return np.array(temp)

def upSampleData(x, y, upSampleFactor = 10, error=None, ext=3):
  upSampX = np.linspace(x[0], x[-1], len(x)*upSampleFactor)

  weights = None
  if error is not None:
    weights = 1/error

  interpolation = interpolate.splrep(x, y, weights)
  upSampY = interpolate.splev(upSampX, interpolation, ext = ext)

  return upSampX, upSampY

def findCenterOfPeak(x,y, peak_half_width = 10):
  mid_point = np.argmax(y)

  left_bound  = mid_point - peak_half_width 
  right_bound = mid_point + peak_half_width + 1

  quad_fit = np.polyfit(x[left_bound:right_bound], y[left_bound:right_bound] ,2)

  center = (-quad_fit[1] / (2*quad_fit[0]))

  return center

def polynomialSubtract(data, polynomialOrder, error=None):
  '''
    Removes a best fit polynomial of order polynomialOrder from data
    Works for 1d/2d data. If data is 2d, removes polynomial from each array in data

    Parameters:
      data (1d/2d array): data from which to remove best fit polynomial

      polynomialOrder (int): order of polynomial to remove

      error (array): errors on data to use for weighting in polynomial fit. same shape as data

    Retuns:
      data: input data after polynomial subtraction
  '''
  is1D = (np.ndim(data)==1)

  seq = data
  if is1D:
    seq = [data]
  
  weights = None
  if error is not None:
    weights = 1/error

  result = []
  x = np.arange(np.shape(data)[-1])

  for i in range(len(seq)):
    arr = seq[i]
    w   = weights[i]

    fit_polynomial = np.polyfit(x, arr, order, w=w)
    polynomial = np.polyval(fit_polynomial, x)
    result.append(arr - polynomial)

  if is1D:
    result = result[0]

  return np.array(result)

def interpolateData(data, old_x, new_x, ext=3):
  '''
    Interpolates the data from domain old_x onto domain new_x

    Parameters:
      data (array): Data to be interpolated

      old_x (array): domain over which data is specified

      new_x (array): domain to interpolate data onto

      ext (int):
        Controls the value returned for elements of x not in the interval defined by the knot sequence.

          if ext=0, return the extrapolated value.
          if ext=1, return 0
          if ext=2, raise a ValueError
          if ext=3, return the boundary value.

    Returns:
      interpolated (array): data over domain new_x
  '''

  splineRep = interpolate.splrep(old_x, data)
  interpolated = interpolate.splev(new_x, splineRep, ext=ext)
  return interpolated

def getSpacing(arr):
  return (arr[-1]-arr[0])/(len(arr)-1)

def normalize(d, outRange=[0,1]):
  num = d-np.min(d)
  den = (np.max(d)-np.min(d))/(outRange[1]-outRange[0])
  return (num/den) +outRange[0]

def percStd(data):
  return (np.percentile(data,84) - np.percentile(data,16))/2

def rowNorm(data):
  return data/np.apply_along_axis(percStd,1,data)[:,np.newaxis]
###

#-- Physics
#TODO: 
  # phase func
  # w = w-180 equal to rv = -rv
  # switch to rv = -rv
def getRV(times, t0=0, P=0, w_deg=0, e=0, Kp=1, v_sys=0,
        vectorizeFSolve = False, returnPhase=False, **kwargs
):
  """
  Computes RV from given model, barycentric velocity

  :param t     : Times of Observations
  :param to    : Time of Periastron
  :param P     : Orbital Period
      # t, t0, P must be same units
  :param w_deg : Argument of periastron
      # degrees
  :param Kp     : Planets Orbital Velocity
  :param v_sys : Velocity of System
      # K, v_sys must be same unit
      # Output will be in this unit
  :return: radial velocity
  """
  w = np.deg2rad(w_deg-180)
  mean_anomaly = ((2*np.pi)/P * (times - t0)) % (2*np.pi)

  if returnPhase:
    return mean_anomaly/(2*np.pi)

  if not vectorizeFSolve:
    try:
      E = []
      for m in mean_anomaly:
        kepler_eqn = lambda E: E - e*np.sin(E) - m
        E.append(optimize.fsolve(kepler_eqn, m)[0])
      E = np.array(E)
    except:
      kepler_eqn = lambda E: E - e*np.sin(E) - mean_anomaly
      E = optimize.fsolve(kepler_eqn, mean_anomaly)
  else:
    kepler_eqn = lambda E: E - e*np.sin(E) - mean_anomaly
    E = optimize.fsolve(kepler_eqn, mean_anomaly)

  true_anomaly = np.arctan2(np.sqrt(1-e**2) * np.sin(E), np.cos(E)-e)
  # return true_anomaly/(2*np.pi)

  # TODO
  # velocity = Kp * (np.cos(true_anomaly+w) + e*np.cos(w)) + v_sys
  velocity = Kp * (np.cos(true_anomaly+w) + e*np.cos(w))

  return velocity

def getBarycentricCorrection(times, starname, obsname, verbose=False, **kwargs):
  if verbose:
    print('Collecting Barycentric velocity.')

  bc=[]
  for time in times:
    JDUTC = Time(time, format='jd',scale='utc')
    output=  barycorrpy.get_BC_vel(JDUTC, starname=starname, obsname=obsname)
    bc.append(output[0][0])

  bc = np.array(bc)
  # Subtract BC to be in target frame
  bc = -bc 
  return bc

def doppler(wave, v, wavelengthFrame='observed', unitPrefix=1):
  '''
    Calculates the doppler shift of a wavelength or array of wavelengths given a relative velocity.

    Wavelength can be entered as either the observed wavelengths, or the source wavelengths.
    Default is observed. If wavelengths are the source wavelength, enter wavelengthFrame='Source'

    Returns Sqrt[(1-beta)/(1+beta)] * wave for observed frame
            Sqrt[(1+beta)/(1-beta)] * wave for source frame

    Parameters:
      wave (float or array): wavelengths to calculate doppler shift of

      v (float): velocity of source relative to observer

      wavelengthFrame (optional): whether the wavelengths are given as the observed wavelengths, or the source wavelengths.
        Options:
          'observed': Wavelengths are observed wavelengths
          'source'  : Wavelengths are source frame wavelengths

      unitPrefix (float): Units of velocity divided by meter/second. 
        i.e. unitPrefix = 1000 implies velocity is in km/s
             unitPrefix = (1000 / 86400) implies velocity is km/day
    Returns:
      correctedWavelengths (float or array): The wavelengths seen in the other frame
  '''
  # Get speed of light in correct units
  c = constants.c / unitPrefix

  # Adjust for wavelengthFrame
  # Note that entering observed wavelengths is the same as entering source wavelengths by with 
  # the relative velocity opposite.
  if wavelengthFrame == 'source':
    v = v
  elif wavelengthFrame == 'observed':
    v = -v
  else:
    raise ValueError('wavelengthFrame must be either "observed" or "source".')

  # Calculate relative difference
  beta = v/c
  xsq = (1+beta)/(1-beta)
  correctedWavelengths = np.sqrt(xsq)*wave

  return correctedWavelengths

def inverseDoppler(observedWave, sourceWave, unitPrefix=1):
  '''
    Given observed and source wavelengths, calculates the relative velocity between the two

    Uses 1+z = wave_obs/wave_source = Sqrt[(1+beta)/(1-beta)]

    Parameters:
      observedWave (float or array): Wavelength in observers frame

      sourceWave (float or array): Wavelength in source frame

      unitPrefix (float): Units of velocity divided by meter/second.
        i.e. unitPrefix = 1000 implies velocity is in km/s
             unitPrefix = (1000 / 86400) implies velocity is km/day
    Returns:
      v (float or array): Veloctiy of source relative to observer (positive = receding)
  '''
  # A = (1+z)^2
  A = (observedWave/sourceWave)**2
  beta = (A - 1)/(A + 1)

  # calculate c in correct units
  c = constants.c / unitPrefix
  v = beta * c

  return v