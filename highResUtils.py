import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

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
          showPlots=False, figsize=(12,8), figTitle="", **kwargs
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

      showPlots (bool): Set true to show the cuts made

      figsize (tuple of 2 integers): figure size for showPlots

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

  if showPlots:
    fig, axs = plt.subplots(2,1,figsize=figsize)
    fig.suptitle(figTitle, size=16)
    axs[0].set_title('Row wise SNR (mean/std) \n After hard cuts, before bounds')
    axs[0].plot(np.apply_along_axis(snr,1,flux))
    axs[0].set_xlabel('Row Number')
    axs[0].set_ylabel('SNR')
  else:
    axs=[None,None]

  if doAutoTrimCols:
    leftEdge, rightEdge = colTrimFunc(flux, plotResult=showPlots, ax=axs[1], **kwargs)

    flux = flux[...,leftEdge:rightEdge]
    if applyColCuts is not None:
      for i in range(len(applyColCuts)):
        applyColCuts[i] = applyColCuts[i][...,leftEdge:rightEdge]
    if applyBothCuts is not None:
      for i in range(len(applyBothCuts)):
        applyBothCuts[i] = applyBothCuts[i][...,leftEdge:rightEdge]

  if showPlots:
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
             returnOffset = False, verbose = False):
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
###

