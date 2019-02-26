import pickle
import numpy as np

from scipy import ndimage as ndi
from scipy import optimize, interpolate

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

def doppler(wave,v, source=False):
  # v in m/s
  # source = False: wave is observed wavelengths
  # source = True: wave is source wavelengths
  beta = v/constants.c
  if source:
    xsq = (1+beta)/(1-beta)
  else:
    xsq = (1-beta)/(1+beta)
  return np.sqrt(xsq)*wave

def inverseDoppler(wave, wave_shift, source=False):
  waveCenter = np.median(wave)
  
  z = wave_shift/ (waveCenter - wave_shift)
  if source:
    z = wave_shift/ (waveCenter)
  A = (1+z)**2
  return (A-1)/(A+1) * constants.c
###