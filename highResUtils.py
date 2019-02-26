import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import signal, interpolate

from utility import *

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

'''
  This package provides functions for doppler spectroscopy that are generic to input data. 

  Written to be used in conjunction with hrsObs.py, which is a class for specific observations and applys these functions
'''

#-- Processing Data
'''
  These functions are for processing the raw data.
'''
def findEdgeCuts_xcor(flux, neighborhood_size=30, 
                      plotResult=False, ax=None, **kwargs
):
  '''
    Detects 'edges' in flux as regions where time-wise SNR dramatically drops off from the center. 

    Find edges by cross correlating snr with a step function and getting first minima/last maxima. 

    Parameters:
      flux (array): 2d array of flux as (time, wavelength)

      neighborhood_size (int): the region around each point to smooth snr/search for extrema

      plotResult (bool): if True, plot the cuts to visualize

      ax (Matplotlib Axes Obj): if set, will plot result on ax

    Returns:
      left_bound, right_bound (int) - the indexes of the found 'edges'
  '''
  # Calculate column (time)-wise SNR in flux
  col_snr = np.nan_to_num(np.apply_along_axis(snr,0,flux))
  col_snr = col_snr - np.mean(col_snr)

  # Apply minimum filter
  smooth = ndi.minimum_filter(col_snr, neighborhood_size)

  # Create step function and cross-correlate with data
  n = int(len(smooth)/2)
  step = np.concatenate((np.ones(n),-1*np.ones(n)))
  xcor = np.correlate(smooth-np.mean(smooth), step, 'same')

  # return first minima - step up in col_snr in begining
  # return last  maxima - step down in col_snr at end
  xcorMinima = getLocalMinima(xcor, neighborhood_size)
  xcorMaxima = getLocalMaxima(xcor, neighborhood_size)
  left_bound  = xcorMinima[0]
  right_bound = xcorMaxima[-1]

  if plotResult:
    # Plot snr, xcor, and cuts
    doShow=False
    if ax is None:
      plt.figure()
      ax = plt.gca()
      doShow=True

    norm_snr = normalize(col_snr)
    norm_smooth = normalize(smooth)

    ax.plot(norm_snr-np.median(norm_snr),label='Column SNR')
    ax.plot(norm_smooth - np.median(norm_smooth),label='Minimum Filter')
    ax.plot(normalize(xcor, (-0.5,0)),label='Cross Correlation')
    ax.plot(normalize(np.median(flux,0),(-1,-0.5)), label='Median Flux')
    ax.plot((left_bound,left_bound),(-1.0,0), color='C2')
    ax.plot((right_bound,right_bound),(-1.0,0), color='C2')

    ax.legend()

    ax.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Normalized SNR')

    if doShow:
      plt.show()

  return left_bound, right_bound

def findEdgeCuts_gradient(flux, gaussian_blur = 10, neighborhood_size = 30,
                          plotResult=False, ax=None, **kwargs
):
  '''
    Detects 'edges' as regions where second derivitive is minimized just inside where the derivitive is minimized (right) or maximized (left). 

    Parameters:
      flux (array): 2d array of flux as (time, wavelength) 

      gaussian_blur (int) : the sigma to use when applying a gaussian filter to the data

      neighborhood_size (int): the region around each point to smooth snr/search for extrema

      plotResult (bool): if True, plot the cuts to visualize

    Returns:
      leftCorner, rightCorner (int) - the indexes of the found 'edges'
  '''

  # Smooth and get gradeint of median spectrum
  signal = np.median(flux,0)
  smooth = ndi.gaussian_filter(signal, gaussian_blur)
  grad = np.gradient(smooth)

  # Find minima on left, maxima on right -> indicate where data drops off the fastest 
  # This is the middle point of the edge
  maxima = getLocalMaxima(grad,neighborhood_size)
  minima = getLocalMinima(grad, neighborhood_size)

  # remove points that are both minima and maxima
  minima_store = minima
  minima = np.setdiff1d(minima,maxima)
  maxima = np.setdiff1d(maxima,minima_store)

  # remove points where minima/maxima = 0
  minima = minima[np.logical_not(np.isclose(grad[minima],0))]
  maxima = maxima[np.logical_not(np.isclose(grad[maxima],0))]

  leftEdge  = maxima[0]
  rightEdge = minima[-1]

  # Find minima of second derivative just inside of the walls found above
  # this is an indicator of the corners - where the wall starts
  secondGrad = np.gradient(grad)
  secondMinima = getLocalMinima(secondGrad,neighborhood_size)

  rightDelta = rightEdge - secondMinima
  rightCorner = rightEdge - np.min(rightDelta[rightDelta>0])

  leftDelta  = secondMinima - leftEdge
  leftCorner = np.min(leftDelta[leftDelta>0]) + leftEdge

  if plotResult:
    doShow=False
    if ax is None:
      plt.figure()
      ax = plt.gca()
      doShow=True

    norm_data = normalize(signal)
    norm_smooth = normalize(smooth)

    ax.plot(norm_data-np.median(norm_data),label='Data')
    ax.plot(norm_smooth - np.median(norm_smooth),label='Smoothed')
    ax.plot((leftCorner,leftCorner),(-0.5,0.5), color='C3')
    ax.plot((rightCorner,rightCorner),(-0.5,0.5), color='C3')

    ax.legend()

    ax.set_title('Edge Trimming\nLeft: '+str(leftCorner)+', Right: '+str(rightCorner))
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Normalized Flux')

    if doShow:
      plt.show()

  return leftCorner, rightCorner

def findEdgeCuts_numeric(flux, edge=0, rightEdge=None, relative=True,
                         plotResult=False, ax=None, **kwargs
):
  '''
    Finds locations to cut the edges of data based on a numeric value entered

    Parameters:
      flux (array): 2d array of flux as (time, wavelength)

      edge (int): how many points to trim from either side

      rightEdge (int): (optional) If entered, will use edge to trim from the left, rightEdge to trim from the right

      relative (bool): (optional) If true, cuts are relative to length of data (i.e. edge = 10 takes 10 points from each side). If false, cuts are absolute (i.e. edge = 10, rightEdge = 1100 cuts between 10 and 1100)
        
      plotResult (bool): if True, plot the cuts to visualize

    Returns:
      left_bound, right_bound (int) - the indexes of the found 'edges'
  '''

  n = np.shape(flux)[1]

  left_bound = edge

  if rightEdge is None:
    if relative:
      right_bound = n-edge
    else:
      raise ValueError('rightEdge must be specified for non-relative cuts.')
  else:
    if relative:
      right_bound = n-rightEdge
    else:
      right_bound = rightEdge

  if plotResult:
    doShow=False
    if ax is None:
      plt.figure()
      ax = plt.gca()
      doShow=True

    normData = normalize(np.median(flux,0))
    
    ax.plot(normData-np.median(normData),label='Median Spectrum')
    ax.plot((left_bound,left_bound),(-0.5,0.5), color='C2')
    ax.plot((right_bound,right_bound),(-0.5,0.5), color='C2')

    ax.legend()

    ax.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
    ax.set_xlabel('Column Number')
    ax.set_ylabel('Normalized Flux')

    if doShow:
      plt.show()

  return left_bound, right_bound

def trimData(flux,
          applyRowCuts=None, applyColCuts=None, applyBothCuts=None,
          rowCuts=None, colCuts=None, doAutoTrimCols=False, colTrimFunc=findEdgeCuts_xcor,
          plotResult=False, figsize=(12,8), figTitle="", **kwargs
):
  '''
    Applys cuts to an array of data

    Parameters:
      flux (array): 2d array of flux as (time, wavelength)

      applyRowCuts (list of arrays): List of arrays to apply row Cuts to (cuts on 0th axis)

      applyColCuts (list of arrays): List of arrays to apply col Cuts to (cuts on -1th axis)

      applyBothCuts (list of arrays): List of arrays to apply both col and row cuts to

      rowCuts (list of integers): Indicies of rows to remove from the data

      colCuts (list of integers): Indicies of columns to remove

      doAutoTrimCols (bool): whether or not to use an automatic column trimming method

      colTrimFunc (function): function to use to autoTrim cols. Requires doAutoTrimCols=True.
                              Options (extra parameters):
                                findEdgeCuts_xcor (neighborhood_size)
                                findEdgeCuts_gradient (gaussian_blur, neighborhood_size)
                                findEdgeCuts_numeric (edge, rightEdge, relative)

      plotResult (bool): Set true to show the cuts made

      figsize (tuple of 2 integers): figure size for plotResult

      figTitle (str): Title for figure

      **kwargs : parameters for colTrimFunc
    
    Returns:
      flux, applyRowCuts, applyColCuts, applyBothCuts: Each of the input datasets after trimming

  '''
  nRows, nCols = flux.shape

  # Apply hard row cuts
  if rowCuts is not None:
    rowMask = np.ones(nRows)
    rowMask[rowCuts] = 0
    rowMask = rowMask.astype(bool)

    flux = flux[rowMask,...]
    if applyRowCuts is not None:
      for i in range(len(applyRowCuts)):
        applyRowCuts[i] = applyRowCuts[i][rowMask,...]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][rowMask,...]

  # Apply hard column cuts
  if colCuts is not None:
    colMask = np.ones(nCols)
    colMask[colCuts] = 0
    colMask = colMask.astype(bool)

    flux = flux[...,colMask]
    if applyColCuts is not None:
      for i in range(len(applyColCuts)):
        applyColCuts[i] = applyColCuts[i][...,colMask]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][...,colMask]

  if plotResult:
    fig, axs = plt.subplots(2,1,figsize=figsize)
    fig.suptitle(figTitle, size=16)
    axs[0].set_title('Row wise SNR (mean/std) \n After hard cuts, before bounds')
    axs[0].plot(np.apply_along_axis(snr,1,flux))
    axs[0].set_xlabel('Row Number')
    axs[0].set_ylabel('SNR')
  else:
    axs=[None,None]

  if doAutoTrimCols:
    leftEdge, rightEdge = colTrimFunc(flux, plotResult=plotResult, ax=axs[1], **kwargs)

    flux = flux[...,leftEdge:rightEdge]
    if applyColCuts is not None:
      for i in range(len(applyColCuts)):
        applyColCuts[i] = applyColCuts[i][...,leftEdge:rightEdge]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][...,leftEdge:rightEdge]

  if plotResult:
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

  return flux,applyRowCuts, applyColCuts,applyBothCuts


def getHighestSNR(flux, error):
  '''
    Gives the index of the spectrum with the hightest median SNR in flux as flux/error

    Parameters:
      flux (array): 2d array of flux as (time, wavelength)

      error (array): array of errors on flux

    Returns:
      highestSNR (int): index of spectrum with highest SNR in flux
  '''
  snrs = np.median(flux/error, 1)
  return np.argmax(snrs)

def calcCorrelationOffset(corr, ref_corr,
                      peak_half_width = 3,
                      upSampleFactor  = 1000,
                      fourier_domain  = False,
                      n               = None,
                      verbose         = False
):
  '''
    Returns the correlation function offsets of corr relative to ref_corr.
    Calculates the location of the peak of the Xcor function by selecting a region around the peak and 
    Interpolating to upsample that region. Then give the center of the upsampled peak

    Parameters:
      corr (2d - array): array of cross correlation functions to find the offsets of

      ref_corr (1d - array): cross correlation function to use as a reference - 
          the center of the peak of this is considered the true zero point

      peak_half_width (int): number of points to include in a region around the xcor peak when upsampling

      upSampleFactor (int): factor by which to upsample the data when interpolating. Limits the precision 
          of the returned centers (i.e. an upSampleFactor of 10 can find peaks to a 0.1 precision level)

      fourier_domain (bool): whether or not corr is an array of fourier transformed correlation functions

      n (int): the length of the original reference array for calculating inverse fft 

      verbose (bool): anounce actions if True

    Returns:
      offsets (1d - array): values of offsets for each row of corr
  '''
  if fourier_domain:
    corr     = ifftCorrelation(corr, n)
    ref_corr = ifftCorrelation(ref_corr, n)


  # Calculate the reference position
  # done by upsampling the peak
  zero_point = np.argmax(ref_corr)
  ref_lb = zero_point - peak_half_width
  ref_rb = zero_point + peak_half_width + 1

  ref_x = range(ref_lb,ref_rb)
  ref_peak = ref_corr[ref_lb:ref_rb]

  upSampX, upSampPeak = upSampleData(ref_x, ref_peak, upSampleFactor=upSampleFactor)
  zero_point =  upSampX[np.argmax(upSampPeak)]

  seq = range(len(corr))
  if verbose:
    seq = tqdm(seq, desc='Calculating Correlation Offsets')

  centers = []
  # Isolate the peak of each Xcor Func and find it's center
  for i in seq:
    xcor = corr[i]
    mid_point = np.argmax(xcor)

    # upsample the peak of the CrossCorrelation function
    xcor_lb = mid_point - peak_half_width
    xcor_rb = mid_point + peak_half_width + 1

    # Record the x-positions and values of the Xcor peak
    peak_x = range(xcor_lb,xcor_rb)
    peak = xcor[xcor_lb:xcor_rb]


    upSampX, upSampPeak = upSampleData(peak_x, peak, upSampleFactor=upSampleFactor)

    center = upSampX[np.argmax(upSampPeak)]
    centers.append(center)

  # Gives the difference between the reference center, and each calculated center
  offsets = zero_point - np.array(centers)

  return offsets

def alignment(flux, ref, iterations = 1,
             error=None, padLen = None,
             peak_half_width = 3, upSampleFactor = 1000,
             returnOffset = False, verbose = False
):
  '''
    Aligns the spectra in flux to the reference spectrum ref. 
    Performs alignments iteratively as commanded. 

    Finds offset by calculating the cross correlation with the reference and finding the peak location

    Applys offset to flux and error and returns it

    Parameters:
      flux (2d-array): 2d array of flux as (time, wavelength)

      ref (1d-array): Reference spectrum 

      iterations (int): Number of times to perform alignment 

      error (2d-array): errors on flux to also align

      padLen (int): amount of zeros to pad to array before fft 

      peak_half_width (int): number of points to include in a region around the xcor peak when upsampling

      upSampleFactor (int): factor by which to upsample the data when interpolating. Limits the precision 
          of the returned centers (i.e. an upSampleFactor of 10 can find peaks to a 0.1 precision level)

      return Offset (bool): If true, returns the offsets, not the shifted flux

    Returns:
      shifted_flux (2d-array): Flux after alignment correction

      shifted_error (2d-array): Error after alignment correction
  '''
  if iterations <=0:
    if error is not None:
      return flux, error
    return flux

  if verbose and not returnOffset:
    print(str(iterations) + ' alignment iterations remaining.')

  m,n = np.shape(flux)

  # Mean subtract for cross correlation
  row_means = np.mean(flux, 1, keepdims=True)
  flux = flux - row_means
  ref  = ref  - np.mean(ref)

  # Pad data with zeros, helps with fft correlation
  # Limits the issues with circular correlation
  # See https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
  if padLen is None:
    padLen = int(n/2)
  ref  = np.pad(ref, padLen, 'constant')
  flux = np.pad(flux, ((0,0),(padLen,padLen)), 'constant')

  # fft_n is the length of the fft, after it has been padded up to length of a power of 2
  ref_fft = rfft(ref)
  flux_fft, fft_n = rfft(flux, returnPadLen=True)

  #correlate the data
  flux_corr_fft = correlate(flux_fft, ref_fft, fourier_domain=True)
  ref_corr_fft  = correlate(ref_fft, ref_fft, fourier_domain=True)

  offsets = calcCorrelationOffset(flux_corr_fft, ref_corr_fft,
                                fourier_domain=True, n=len(ref), 
                                peak_half_width=peak_half_width,
                                upSampleFactor=upSampleFactor,
                                verbose=verbose)

  if returnOffset:
    return offsets

  # Apply shifts to data
  shifted_fft = fourierShift2D(flux_fft, offsets, n=fft_n, fourier_domain=True)
  if error is not None:
    shifted_error = fourierShift2D(error, offsets, fourier_domain=False)
  else:
    shifted_error = None

  # Un-fft and remove padding
  # replace means
  shifted_flux = np.fft.irfft(shifted_fft)[:,padLen:n+padLen] + row_means
  return alignment(shifted_flux, ref, iterations=iterations-1,
                    error=shifted_error, padLen=padLen, peak_half_width=peak_half_width,
                    upSampleFactor=upSampleFactor, verbose=verbose)

def getTimeMask(flux, relativeCutoff=3, absoluteCutoff=0,
                smoothingFactor=0, plotResult=False
):
  '''
    Calculates a boolean mask based on which columns in flux have low SNR.
    Low SNR is defined as a SNR of (relativeCutoff) sigma below the mean SNR,
    or as an SNR below (absoluteCutoff)

    Parameters:
      flux (2d-array): 2d array of flux as (time, wavelength)

      relativeCutoff (positive float): Mask columns with SNR this sigma below the mean SNR

      absoluteCutoff (float): Mask columns with SNR below this value

      smoothingFactor (int): Number of columns around a masked column to also mask

    Returns:
      mask (1d-array): Boolean array of 1's for good columns, 0's for bad columns
  '''

  # Calculate SNRs
  weights = np.nan_to_num(np.apply_along_axis(snr, 0, flux))

  if np.any(weights < 0):
    print('Warning, some weights (SNRs) are less than zero, consider using non zero-mean values for generating mask')

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  # combine the masks and make it so 1 is good, 0 is bad
  mask = 1 - np.logical_or(lowerMask, absoluteMask)

  # widen the mask by smoothing factor
  mask = ndi.minimum_filter(mask, smoothingFactor)

  if plotResult:
    plt.figure(figsize=(12,8))
    plt.subplot(211)
    plt.title('Column wise SNRs')
    plt.plot(weights)
    n = len(weights)
    colors = ['C1','C2','C3','C4']

    # Plot sigma lines
    labels = ['Mean','1 sigma','2 sigma','3 sigma']
    for i in range(-3,4):
      ls = '-'
      lab = labels[np.abs(i)]
      if i < 0:
        lab = ""
      plt.plot((0,n),(weightMean+i*weightStd,weightMean+i*weightStd),
        label=lab, linestyle=ls, color=colors[np.abs(i)])

    # plot absolute cutoff
    plt.plot((0,n),(absoluteCutoff,absoluteCutoff),label='Absolute Cutoff',
      linestyle='--',color='k')

    plt.legend(frameon=True,loc='best')

    # Show full mask
    plt.subplot(212)
    plt.title('Time Mask')
    plt.plot(normalize(np.median(flux,0)))
    plt.plot(mask)
    plt.ylim(-0.2,1.2)
    plt.show()

  return mask

def getWaveMask(flux, windowSize=25, relativeCutoff=3,
                absoluteCutoff=0, smoothingFactor=0,
                plotResult=False
):
  '''
    Calculates a boolean mask based on the SNR of a region around each column. This method finds regions of sharp change in flux which correspond to deep telluric absorbtion.

    Regions of low SNR are masked
    Low SNR is defined as a SNR of (relativeCutoff) sigma below the mean SNR,
    or as an SNR below (absoluteCutoff)

    Parameters:
      flux (2d-array): 2d array of flux as (time, wavelength)

      windowSize (int): Size of region around each wavelength column to consider for calculating SNR

      relativeCutoff (positive float): Mask columns with SNR this sigma below the mean SNR

      absoluteCutoff (float): Mask columns with SNR below this value

      smoothingFactor (int): Number of columns around a masked column to also mask

    Returns:
      mask (1d-array): Boolean array of 1's for good columns, 0's for bad columns
  '''

  # Calulate weights
  medSpec = np.median(flux, 0)
  weights = ndi.generic_filter(medSpec, snr, size=windowSize)

  weightMean = np.mean(weights)
  weightStd  = np.std(weights)

  # create masks
  lowerMask = weights < weightMean - relativeCutoff*weightStd
  absoluteMask = weights < absoluteCutoff

  # combine the masks and make it so 1 is good, 0 is bad
  mask = 1 - np.logical_or(lowerMask, absoluteMask)

  # widen the mask by smoothing factor
  mask = ndi.minimum_filter(mask, smoothingFactor)

  if plotResult:
    plt.figure(figsize=(6,8))
    plt.subplot(211)
    plt.title('Windowed SNR along row')
    plt.plot(weights)
    n = len(weights)
    colors = ['C1','C2','C3','C4']

    # Plot Relative cutoffs
    labels = ['Mean','1 sigma','2 sigma','3 sigma']
    for i in range(-3,4):
      ls = '-'
      lab = labels[np.abs(i)]
      if i < 0:
        lab = ""
      plt.plot((0,n),(weightMean+i*weightStd,weightMean+i*weightStd),
        label=lab, linestyle=ls, color=colors[np.abs(i)])

    # Plot Rbsolute cutoffs
    plt.plot((0,n),(absoluteCutoff,absoluteCutoff),label='Absolute Cutoff',
      linestyle='--',color='k')

    plt.legend(frameon=True)
    
    plt.subplot(212)
    plt.title('Wave Mask')
    plt.plot(normalize(np.median(flux,0)))
    plt.plot(mask)
    plt.ylim(-0.2,1.2)
    plt.show()

  return mask

def combineMasks(*masks, smoothingFactor=20):
  '''
    Combines the input binary masks and widens each mask element by smoothingFactor

    Parameters:
      masks (1D array): Binary arrays to be combined with one another

      smoothingFactor (int): Amount to widen each mask element by
    Returns:
      combinedMask (1D array): Binary array of combined masks
  '''
  mask = np.prod(masks,0)
  return ndi.minimum_filter(mask, smoothingFactor)

def applyMask(data, mask):
  '''
    Applys binary mask to data and subtracts mean from data row-wise.
    Applys mask such that after subtracting the mean, masked regions have a zero value.

    Parameters:
      data (2d-array): 2d array of data

      mask (1d-array): Binary array of which columns of data to mask

    Returns:
      masked (2d-array): data with row-means subtracted and masked columns set to zero-value
  '''

  # Calculates what the mean will be excluding masked columnns
  numUnmasked = np.sum(mask)
  newMean = np.sum(data*mask, -1, keepdims=1)/numUnmasked

  # Subtracts the new mean and applys mask
  # Rows of masked now have zero mean, masked regions have zero value
  masked = data - newMean
  masked = masked * mask

  return masked

def sysrem(data, error, ncycles=1,
           initialGuess=None,
           maxIterations=200,
           maxError=0.001,
           verbose=False,
           returnAll=True
):
  '''
    Implementation of the Sysrem de-trending algorithm from (Tamuz+ 2005). See also (Mazeh+ 2007). Removes systematic effects from many lightcurves. A variant of PCA with non-equal errors.

    Given a 2d-array of data, removes linear trends in the vertical (along columns) direction.

    Removes the trends found and returns either the residual data, or the list of succsessive residuals after each cycle

    Parameters:
      data (2d-array): Data aranged so each column represents an "independent lightcurve". Can be used so that
          each column represents a wavelength channel and each row a spectrum
      error (2d-array): error on the values in data

      ncycles (positive int): Number of cycles to run sysrem. Analogous to number of Prinicpal Components to
          remove

      initalGuess (vector): Inital guess of trend to remove

      maxIterations (int): How many iterations to attempt on fitting each trend before aborting

      maxError (float): Maximum amount a found trend can vary by before it is considered stationary (and the
          algorithm converged)

      verbose (bool): Print progress of algorithm

      returnAll (bool): Whether to return the final residuals, or each set of residuals after each cycle

    Returns:
      residuals (2d-array): data after linear trends are removed
  '''

  # Sysrem works on N lightcurves each of M points
  # Analogous to N wavelength channels of M observations in each
  # Assume data is passed so each row is a spectrum or each column is a lightcurve
  M,N = np.shape(data)

  # Subtract the mean from each column
  # subtracts mean from each lightcurve
  residuals = data - np.mean(data,0)

  allResiduals = [residuals]

  # Set initial guess if none passed
  if initialGuess == None:
    initialGuess = np.ones(M)

  invErrorSq = 1/(error**2)

  # Initialize a and c
  # a is a trend constant at a single time (e.g. airmass)
  # c is a trend constant for each lightcurve (e.g. extinction coefficient)
  aVec = initialGuess
  cVec = np.ones(N)

  if verbose:
    print('Starting Sysrem')

  for cycle in range(ncycles):
    if verbose:
      print('Starting Cycle '+str(cycle+1),flush=True)
      pbar = tqdm(total=100, desc='Cycle '+str(cycle+1))

    aVecError = maxError * 10
    cVecError = maxError * 10

    iterations = 0

    # Succsessively calculate an a and c from the inital guess until both converge
    # When converged, we have found the a,c that work for Tamuz+ 2005 Eq 1
    while iterations <= maxIterations and (aVecError >= maxError or cVecError >= maxError):
      # Store last a and c for calculating difference
      last_aVec = aVec
      last_cVec = cVec

      # Tamuz+ 2005 Eq 2
      cVecNum = np.sum( (residuals * aVec[:,np.newaxis]) * invErrorSq, 0)
      cVecDen = np.sum( ((aVec**2)[:,np.newaxis])        * invErrorSq, 0)
      cVec = cVecNum/cVecDen

      # Tamuz+ 2005 Eq 4
      aVecNum = np.sum( residuals * cVec * invErrorSq, 1)
      aVecDen = np.sum( cVec**2          * invErrorSq ,1)
      aVec = aVecNum/aVecDen

      # Calculate difference from last a,c for convergence
      aVecError = np.median(np.nan_to_num(np.abs(last_aVec / aVec -1 )))
      cVecError = np.median(np.nan_to_num(np.abs(last_cVec / cVec - 1 )))

      if verbose:
        largestError = np.max((aVecError, cVecError))
        convergence = 1/(np.log(largestError/maxError)+1)
        if largestError <= maxError:
          convergence = 1

        pbarVal = int(convergence*100)
        pbar.update(max(pbarVal-pbar.n, 0))

      iterations += 1
    # a,c have converged
    # we have found the best fit trends for this cycle of sysrem
    # model the data from these trends
    thisModel = np.outer(aVec,cVec)

    # calculate new set of residuals
    residuals = residuals - thisModel
    allResiduals.append(residuals)

  # Finished full set of sysrem cycles
  if returnAll:
    return np.array(allResiduals)
  else:
    return allResiduals[-1]

def varianceWeighting(data, axis=-2):
  '''
    Weights the data by the variance along axis, default is -2

    For data.ndim = 2: Uses the column variance
    For data.ndim = 3: Uses the column variance of each image

    Parameters:
      data (2+ D array): Input data to weight by variance

      axis (int): axis along which to apply variance weighting

    Returns:
      weighted (array): Data after variance weighting is applied
  '''
  
  return np.nan_to_num(data/np.var(data,-2, keepdims=1))
###

#-- Comparing Data to template
def generateXCM(data, template,
                normalizeXCM=True,
                xcorMode='same',
                verbose=False
):
  '''
    Caclulates the cross-correlation of each row of data versus template. Assembles the resultant
    cross correlation functions into a matrix and returns result.

    Parameters:
      data (2d-array): 2d array of processed spectra. Each row is a spectrum.

      template (1d-array): template spectrum against which to cross correlate the data

      normalizeXCM (bool): whether or not to normalize the cross correlation functions according to Zucker 2003

      xcorMode (string): Mode of cross correlation, passed to scipy.signal.correlate.
                Options:
                  same
                    The output is the same size as each spectra in data, centered with respect to the ‘full’ output.
                    (Default)

                  full
                    The output is the full discrete linear cross-correlation of the inputs.

                  valid
                    The output consists only of those elements that do not rely on the zero-padding. In ‘valid’ mode, either each spectra in data or template must be at least as large as the other in every dimension.
      verbose (bool): Creates progress bar if true

    Returns:
      xcm (2d-array): matrix where each row is the corresponding row of data cross correlated against template
  '''
  # Remove mean from cross-correlation inputs
  template = template - np.mean(template)
  data     = data     - np.mean(data,1)[:,np.newaxis]

  seq = data
  if verbose:
    seq = tqdm(seq, desc='Cross Correlating')

  xcm = []
  for spec in seq:
    xcor = signal.correlate(spec, template, xcorMode)

    if normalize:
      n = len(spec)
      #Perscription in Zucker 2003
      xcor = xcor / (n*np.std(spec)*np.std(template))

    xcm.append(xcor)

  return np.array(xcm)

def alignXCM(xcm, xcorVels, velocityOffsets,
              isInterpolatedXCM=False, ext=1
):
  '''
    Shifts each cross correlation function in xcm according to a velocity offset in (velocityOffsets)

    Parameters:
      xcm (2d-array): Matrix where each row is a cross correlation function (CCF)

      xcorVels (1d-array): velocity axis for each CCF

      velocityOffsets (1d-array): velocity offset for each individual CCF.
        Length must be equal to the length of xcm (number of CCFs in xcm)

      isInterpolatedXCM (bool): Allows one to set xcm as the output of interpolate.splrep(xcorVels, xcm)
        Usefull for faster computation if run many times.

      ext (int): Controls the value returned for elements of x not in the interval defined by the knot sequence.
          if ext=0, return the extrapolated value.
          if ext=1, return 0
          if ext=2, raise a ValueError
          if ext=3, return the boundary value.
    Returns:
      alignedXCM (2d-array): the input xcm with each CCF shifted by a velocity offset
  '''

  # If not already interpolated, make interpolation matrix
  if not isInterpolatedXCM:
    xcm = [interpolate.splrep(xcorVels, xcor) for xcor in xcm]

  # Assemble the modified velocities
  adjustedVelocities = [xcorVels + velocityOffset for velocityOffset in velocityOffsets]

  # Align the xcm according to the new velocities
  alignedXCM = [interpolate.splev(adjustedVelocities[i], xcm[i], ext=ext) for i in range(len(xcm))]

  return np.array(alignedXCM)
###