import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

from utility import *
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

###

