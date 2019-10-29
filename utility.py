import pickle
import numpy as np
import os
import copy

from scipy import ndimage as ndi
from scipy import optimize, interpolate

from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.modeling.blackbody import blackbody_lambda
from astropy import constants

import barycorrpy
from barycorrpy.utils import get_stellar_data

def type_of_script():
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'

if type_of_script() == 'jupyter':
  from tqdm import tqdm_notebook as tqdm
else:
  from tqdm import tqdm

#-- File I/O
def readFile(dataFile, delimiter=','):
  '''
    Reads data from file. Currently supports fits, pickle, csv

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
  elif extension == 'csv':
    data = np.loadtxt(dataFile,delimiter=delimiter)
  else:
    raise ValueError('Error reading "'+dataFile+'" Currently only fits pickle and csv are supported')

  return data

def makePaths(path):
  '''
    Given a path, this function goes through all included subpaths and creates them if they do not already exist

    Parameters:
      path (str): directory path e.g. "this/is/a/path/"
  '''
  if path[-1] != '/':
    path = path+'/'

  dirList = path.split('/')

  for i in range(1, len(dirList)):
    subPath = '/'.join(dirList[:i])
    try:
      os.mkdir(subPath)
    except FileExistsError:
      pass

  return path

def pickleOpen(fn):
  with open(fn,'rb') as f:
    return pickle.load(f)
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

def createNestedDict(*keyLists, innerDict={}):
  '''
    creates a nested dictionary from a list of keyLists
    each key list creates a layer in the dictionary

    e.g.
    keyLists = (['a','b'], ['1','2'],['#'])

    output:
      { 'a': {
          1: {
            '#': {}
          },
          2: {
            '#': {}
          }
        },
        'b': {
          1: {
            '#': {}
          },
          2: {
            '#': {}
          }
        },
      }
  '''
  

  last = keyLists[-1]
  rest = keyLists[:-1]
  
  outerDict = {}
  for key in last:
    outerDict[key] = copy.deepcopy(innerDict)

  if len(rest) == 0:
    return outerDict

  return createNestedDict(*rest, innerDict=outerDict)

def collapseDict(theDict):
  keys = list(theDict.keys())
  if len(keys) == 1:
    theDict = theDict[keys[0]]
  else:
    return theDict

  return collapseDict(theDict)

def intersection(list1, *args):
  return list(set(list1).intersection(*args))
###

#-- Math
def gaussian(x, mu, sig):
  '''
    returns a gaussian with mean mu, std deviation sig, on domain x

    Parameters:
      x (array): domain for gaussian

      mu (float): mean of gaussian

      sig (float): standard deviation

    Returns:
      y (array): gaussian values on domain x
  '''

  normalization = 1/np.sqrt(2 * np.pi * sig**2)
  exponent = - ((x - mu)**2 / (2*sig**2))
  y = normalization * np.exp(exponent)
  return y

def sigma2fwhm(sigma):
  '''
    converts gaussian sigma to full width at half maximum
  '''
  return sigma * np.sqrt(8 * np.log(2))

def fwhm2sigma(fwhm):
  '''
    converts gaussian full width at half maximum to sigma
  '''
  return fwhm / np.sqrt(8 * np.log(2))
###

#-- Data
def maxDiff(A,B):
  return np.max(np.abs(A-B))

def snr(data):
  ''' 
    Gives the SNR of a data, recommended to be a vector

    Parameters: 
      data (array) : the input array

    Returns:
      snr (float) : the snr of the data
  '''
  return np.mean(data)/np.std(data)

def getOutliers(data, n_sigma=3, twoSided=False):
  '''
    Returns positions of values in data that are n_sigma outliers from the mean value
  '''

  mean = np.mean(data)
  std  = np.std(data)

  direction = np.sign(n_sigma)
  n_sigma = np.abs(n_sigma)

  low_outliers = np.where(data < mean - n_sigma*std)[0]
  high_outliers = np.where(data > mean + n_sigma*std)[0]

  if twoSided:
    return np.concatenate((low_outliers, high_outliers))
  else:
    if direction < 0:
      return low_outliers
    else:
      return high_outliers

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

def upSampleData(x, y, upSampleFactor=10, error=None, ext=3):
  upSampX = np.linspace(x[0], x[-1], len(x)*upSampleFactor)

  weights = None
  if error is not None:
    weights = 1/error

  interpolation = interpolate.splrep(x, y, weights)
  upSampY = interpolate.splev(upSampX, interpolation, ext = ext)

  return upSampX, upSampY

def findCenterOfPeakSpline(data, approximateCenter, maxima=1, width=4,
  upSampleFactor=10
):
  x = np.arange(-width,width+1)
  y = data[x+approximateCenter]
  upSampX, upSampY = upSampleData(x,y, upSampleFactor)

  if maxima==1:
    centerOffset = upSampX[np.argmax(upSampY)]
  else:
    centerOffset = upSampX[np.argmin(upSampY)]

  return centerOffset+approximateCenter

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

def closestArgmin(A, B, edgeBehavior=0):
  '''
    For each element in A, finds the argument of the closest element in B.
    Returns said arguments

    found: https://stackoverflow.com/questions/45349561/find-nearest-indices-for-one-array-against-all-values-in-another-array-python

    Parameters:
      A (array): Array that we want closest elements in B for

      B (array): Array of elements to compare against

      edgeBehavior (int): determines how to handle elements in A outside the range of B
        0 (default): Puts all values in A external to B in the edge bins
        1 : Returns -1 for all values of A below lowest B, len(B) for all values of A above highest B
        2 : Extrapolates B to have one more point in either direction, sorts according to edgeBehavior=0.
            Returns -1 for all values belonging to (new) lowest bin, len(B) for all belonging to (new) highest Bin

    Returns:
      args (array): array of arguments of B elements, which are closest to each A element. 
  '''

  if edgeBehavior==2:
    # Add external bins to B based on typical spacing and highest/lowest values
    dB = getSpacing(B)
    C = np.concatenate(([np.min(B)-dB],B,[np.max(B)+dB]))
    B = C

  L = B.size
  # Sort B to make use of np.searchsorted
  sidx_B = B.argsort()
  sorted_B = B[sidx_B]

  # Finds "right positions" -  B value that is just Above A
  sorted_idx = np.searchsorted(sorted_B, A)

  # If elements are beyond largest in B, return largest in B
  sorted_idx[sorted_idx==L] = L-1

  # determine if we need "left" or "right" positions
  # mask is 1 if closer to "left" value - indicates need to round down
  mask = (sorted_idx > 0) & \
  ((np.abs(A - sorted_B[sorted_idx-1]) < np.abs(A - sorted_B[sorted_idx])) )

  # Get original arguments of desired values
  args = sidx_B[sorted_idx-mask]

  if edgeBehavior==1:
    # External values get marked as external
    args[A < np.min(B)] = -1
    args[A > np.max(B)] = len(B)
  elif edgeBehavior==2:
    args = args-1

  return args

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
        vectorizeFSolve = False, returnPhase=False, returnPhaseAndRV=False,
        inputPhases=False,
        **kwargs
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
  if inputPhases:
    mean_anomaly = times * (2*np.pi)

  else:
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

  if returnPhaseAndRV:
    return velocity, mean_anomaly/(2*np.pi)

  return velocity

def getBarycentricCorrection(times, starname, obsname, verbose=False, **kwargs):
  bc=[]
  if verbose:
    seq = tqdm(times, desc='Collecting Barycentric velocity')
  else:
    seq = times
  for time in seq:
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
  c = constants.c.value / unitPrefix

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
  c = constants.c.value / unitPrefix
  v = beta * c

  return v

def reduceSpectralResolution(x, y, R_low, R_high=None, lambda_mid=None, n=4):
  '''
     Reduces the spectral resolution of the input spectrum from R_high to R_low, by convolution.
     Convolves input spectrum with gaussian kernel of fwhm:
        np.sqrt(d_lambda_low^2 - d_lambda_high^2)

    Parameters:
      x (array): Input spectrum wavelengths

      y (array): Input spectrum Values

      R_low (float): Desired output resolution

      R_high (float, optional): Input spectral resolution, if not entered, will be estimated from input spectrum

      lambda_mid (float, optional): Midpoint of spectrum where R was calculated. If not entered, will be
          estimated from input spectrum

      n (float): Width of gaussian kernel in sigma

    Returns:
      lowRes (array): Reduced resolution spectrum on same wavelength grid as x
  '''
  dx = getSpacing(x)

  # If lambda_mid is none, take median of input wavelengths
  if lambda_mid is None:
    lambda_mid = np.median(x)

  # If R_high is none, use midpoint of x divided by spacing in x
  if R_high is None:
    R_high = lambda_mid/dx

  # Create Gaussian kernel
  fwhm = np.sqrt(R_high**2 - R_low**2)*(lambda_mid/(R_low*R_high))
  sigma = fwhm2sigma(fwhm)

  kernel_x = np.arange(-n*sigma, n*sigma+dx, dx)
  kernel = gaussian(kernel_x,0,sigma)
  kernel = kernel/np.sum(kernel)

  # find center of kernel
  n_kernel_lt0 = len(np.where(kernel_x<0)[0])
  n_kernel_gt0 = len(np.where(kernel_x>0)[0])

  if n_kernel_lt0< n_kernel_gt0:
    origin = 0
  else:
    origin=-1

  # convolve
  lowRes = ndi.convolve(y, kernel, origin=origin)
  return lowRes

def equilibriumTemperature(P, T_star, R_star, M_star):
  '''
    Given planet period, stellar parameters, calculates the equlibrium temperature for the planet.

    Parameters:
      P - days
      T_star - Kelvin
      R_star - Solar Radii
      M_star - Solar Mass

    Returns:
      T_eq - planet equlibrium temperature
  '''

  T_eq = 344.67 * (T_star/1000) * (R_star)**(1/2) * M_star**(-1/6) * P**(-1/3)
  return T_eq

def assertUnits(x, x_units, x_default=None):
  # check if x has units
  try:
    x.unit
  except AttributeError:
    # x needs units

    # if x_units is a unit, put it on
    # hacky way to check if x_units is a unit
    if 'astropy.units.core' in str(type(x_units)):
      x = x * x_units

    # if x_units is numeric, use that as prefix to x_default
    elif isinstance(x_units, int) or isinstance(x_units, float):
      if x_default is None:
        raise ValueError("If x_units is numeric, x_default must be astropy.unit")

      x = x * x_default * x_units

    else:
      raise TypeError("x must have units, or x_unit must be either a int/float/astropy.unit")

  return x



def blackbody(w, T, w_units=u.micron):
  '''
    Returns the spectral radiance per unit wavelength (B_lambda) for a blackbody at temperature T, 
    over wavelengths w. Wrapper for astropy.modelinig.blackbody.blackbody_lambda

    Parameters:
      w (float or array): wavelengths to evalualte B_lambda over

      T (float): Temperature of blackbody

      w_units (float or astropy.units.core.Unit): Units of wavelength
        If float, considered as prefix relative to meters i.e.
          1 - meters
          10e-6 - microns
          10e-9 - nanometers

    Returns:
      B_lambda (same as w): spectral radiance per unit wavelength for input wavelengths
          units: erg cm^-2 s^-1 Angstrom^-1
  '''

  # assert w has units
  w = assertUnits(w, w_units, u.m)

  return blackbody_lambda(w, T)

def getBlackBodyFlux(T, w=None, w_units=u.micron):
  '''
  '''

  # if no wavelength specified, run stefan boltzmann eq
  if w is None:

    # Check if T has units, or else apply kelvin
    try:
      T.unit
    except AttributeError:
      # T needs units

      T = T*u.Kelvin

    flux = constants.sigma_sb * T

  # Wavelength region specified - integrate blackbody radiance times pi
  else:
    # assert first w has units
    w = assertUnits(w, w_units, u.m)

    radiancePerW = blackbody(w, T) #erg / AA / s/ cm^2

    w_AA = w.to(u.Angstrom)
    radiance = np.trapz(radiancePerW, w_AA) #erg s^-1 cm^-2 sr^-1
    flux = radiance * np.pi * u.sr

  return flux

def getBlackBodyLuminosity(T, R, w=None, w_units=u.micron, R_units=u.R_sun):
  '''
  '''

  flux = getBlackBodyFlux(T, w, w_units)

  # Assert units of R
  R = assertUnits(R, R_units, u.R_sun)

  luminosity = flux * np.pi * (R**2)
  return luminosity

###