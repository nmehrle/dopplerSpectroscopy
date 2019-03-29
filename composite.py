import json

from hrsObs import *
from utility import *
import highResUtils as hru

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import interpolate

from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial

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



#-- Composite Functions
def prepareData(obs,
                  # TrimData
                  doAutoTrimCols=True, plotTrim=False,
                  colTrimFunc=hru.findEdgeCuts_xcor,
                  neighborhood_size=30, gaussian_blur=10,
                  edge=0, rightEdge=None, relative=True,
                  #alignData
                  alignmentIterations=1, alignmentPadLen=None,
                  alignmentPeakHalfWidth=3, alignmentUpSampFactor=1000,
                  #injectData
                  doInjectSignal=False, injectedKp=None, injectedVsys=None,
                  injectedRelativeStrength=None, unitPrefix=1000,
                  #normalize
                  normalizationScheme='divide_row',polyOrder=2,
                  #generateMask
                  use_time_mask=True, use_wave_mask=False,
                  plotMasks=False, maskRelativeCutoff=3,
                  maskAbsoluteCutoff=0, maskSmoothingFactor=20,
                  maskWindowSize=25,
                  # sysrem
                  sysremIterations=None,
                  stopBeforeSysrem=False,
                  verbose=False,
                  # Post Sysrem
                  doVarianceWeight=True
):
  '''
    Performs the standard data preperation.
    Use this before calling obs.generateXCM().
  '''

  obs.trimData(doAutoTrimCols=doAutoTrimCols, plotResult=plotTrim,
                colTrimFunc=colTrimFunc,
                neighborhood_size=neighborhood_size, gaussian_blur=gaussian_blur,
                edge=edge, rightEdge=rightEdge, relative=relative)

  obs.alignData(iterations=alignmentIterations, padLen=alignmentPadLen,
                 peak_half_width=alignmentPeakHalfWidth, upSampleFactor=alignmentUpSampFactor,
                 verbose=verbose)

  if doInjectSignal:
    print('---------------------------------')
    print('----- Injecting Fake Signal -----')
    print('---------------------------------')

    obs.injectFakeSignal(injectedKp=injectedKp, injectedVsys=injectedVsys,
                          relativeStrength=injectedRelativeStrength,
                          unitPrefix=unitPrefix, verbose=verbose)

  obs.generateMask(use_time_mask=use_time_mask, use_wave_mask=use_wave_mask, plotResult=plotMasks,
                    relativeCutoff=maskRelativeCutoff, absoluteCutoff=maskAbsoluteCutoff,
                    smoothingFactor=maskSmoothingFactor, windowSize=maskWindowSize)

  obs.normalizeData(normalizationScheme=normalizationScheme, polyOrder=polyOrder)

  obs.applyMask()

  if not stopBeforeSysrem:
    obs.sysrem(nCycles=sysremIterations, verbose=verbose)

    if doVarianceWeight:
      obs.varianceWeight()

    obs.applyMask()

def xcorAnalysis(obs, kpRange,
                  # Generate XCM
                  normalizeXCM=True, xcorMode='same',
                  # Generate SigMat
                  outputVelocities=None,
                  # Normalize SigMat
                  doNormalizeSigMat=True,
                  rowByRow=False, byPercentiles=False,
                  # General
                  unitPrefix=1000, verbose=False
):
  '''
    Performs the steps in the Cross Correlation Analysis
    Call obs.prepareData() then obs.xcorAnalysis()
  '''

  obs.generateXCM(normalizeXCM=normalizeXCM, xcorMode=xcorMode,
                   unitPrefix=unitPrefix, verbose=verbose)

  obs.generateSigMat(kpRange, unitPrefix=unitPrefix, outputVelocities=outputVelocities, verbose=verbose)

  obs.reNormalizeSigMat(rowByRow=rowByRow, byPercentiles=byPercentiles)

def reportOnSingleSysremIteration(obs, iteration, allSysremData, kpRange,
                  # Preparation
                  doVarianceWeight=True,
                  # XCor Analysis KWs
                    # Generate XCM
                    normalizeXCM=True, xcorMode='same',
                    # Generate SigMat
                    outputVelocities=None,
                    # Normalize SigMat
                    rowByRow=False, byPercentiles=False,
                  # Report Detection Strength KWs
                    kpSearchExtent=2, vsysSearchExtent=4,
                    plotDetection=False, savePrefix=None,
                    plotKpExtent=40, plotVsysExtent=40,
                    detectionUnitStr='km/s',
                    showDetectionPlots=False,
                    closeDetectionPlots=True,
                    detectionTitle='',
                    clim=[None,None],
                    cmap='viridis',
                    figsize=None,
                  # General
                    targetKp=None, targetVsys=None,
                    unitPrefix=1000, verbose=False
):
  '''
    Used to allow reportDetectionStrength to be caled in a multiprocessing context
  '''
  theCopy = obs.copy()
  theCopy.data = allSysremData[iteration]

  if doVarianceWeight:
    theCopy.varianceWeight()

  theCopy.applyMask()

  theCopy.xcorAnalysis(kpRange,
                       normalizeXCM=normalizeXCM,
                       xcorMode=xcorMode,
                       outputVelocities=outputVelocities,
                       rowByRow=rowByRow,
                       byPercentiles=byPercentiles,
                       unitPrefix=unitPrefix
  )

  if detectionTitle != '' and detectionTitle[-1] != '\n':
    detectionTitle = detectionTitle + '\n'

  titleStr = detectionTitle + 'Sysrem Iterations: '+str(iteration)
  if savePrefix is None:
    saveName = None
  else:
    saveName = savePrefix + 'sysIt_'+str(iteration)+'.png'

  detStrength, detCoords = theCopy.reportDetectionStrength(targetKp=targetKp,
                                           targetVsys=targetVsys,
                                           kpSearchExtent=kpSearchExtent,
                                           vsysSearchExtent=vsysSearchExtent,
                                           plotResult=plotDetection,
                                           plotKpExtent=plotKpExtent,
                                           plotVsysExtent=plotVsysExtent,
                                           unitStr=detectionUnitStr,
                                           show=showDetectionPlots,
                                           close=closeDetectionPlots,
                                           clim=clim, cmap=cmap,
                                           figsize=figsize,
                                           title=titleStr,
                                           saveName=saveName)

  return detStrength, detCoords, theCopy.unNormedSigMat

def reportSysremIterations(obs, kpRange,
                  maxIterations=10,
                  cores=1,
                  # Prepare Data KWs
                    doAutoTrimCols=True,
                    plotTrim=False,
                    colTrimFunc=hru.findEdgeCuts_xcor,
                    neighborhood_size=30, gaussian_blur=10,
                    edge=0, rightEdge=None, relative=True,
                    #alignData
                    alignmentIterations=1,
                    alignmentPadLen=None,
                    alignmentPeakHalfWidth=3,
                    alignmentUpSampFactor=1000,
                    #injectData
                    doInjectSignal=False,
                    injectedRelativeStrength=1/1000,
                    #normalize
                    normalizationScheme='divide_row',polyOrder=2,
                    #generateMask
                    use_time_mask=True, use_wave_mask=False,
                    plotMasks=False, maskRelativeCutoff=3,
                    maskAbsoluteCutoff=0, maskSmoothingFactor=20,
                    maskWindowSize=25,
                    # After Sysrem
                    doVarianceWeight=True,
                  # XCor Analysis KWs
                    # Generate XCM
                    normalizeXCM=True, xcorMode='same',
                    # Generate SigMat
                    outputVelocities=None,
                    # Normalize SigMat
                    rowByRow=False, byPercentiles=False,
                  # Report Detection Strength KWs
                    kpSearchExtent=2, vsysSearchExtent=4,
                    plotDetection=False, savePrefix=None,
                    plotKpExtent=60, plotVsysExtent=80,
                    detectionUnitStr='km/s',
                    showDetectionPlots=False,
                    closeDetectionPlots=True,
                    detectionTitle='',
                    clim=[None,None],
                    cmap='viridis',
                    figsize=None,
                  # General
                    targetKp=None, targetVsys=None,
                    unitPrefix=1000, verbose=False
):
  '''
    Given a hrsObs object, performs the data preparation and analysis,
    and then reports the detection strength for each sysremIteration up to the
    maxIterations specified.

    Can be done in multiprocessing if cores > 1
  '''

  superVerbose = (verbose>1)

  obs.collectRawData()
  obs.prepareData(stopBeforeSysrem=True,
                    doAutoTrimCols=doAutoTrimCols, plotTrim=plotTrim,
                    colTrimFunc=colTrimFunc,
                    neighborhood_size=neighborhood_size, gaussian_blur=gaussian_blur,
                    edge=edge, rightEdge=rightEdge, relative=relative,
                    #alignData
                    alignmentIterations=alignmentIterations,
                    alignmentPadLen=alignmentPadLen,
                    alignmentPeakHalfWidth=alignmentPeakHalfWidth,
                    alignmentUpSampFactor=alignmentUpSampFactor,
                    #injectData
                    doInjectSignal=doInjectSignal,
                    injectedKp=targetKp, injectedVsys=targetVsys,
                    injectedRelativeStrength=injectedRelativeStrength,
                    #normalize
                    normalizationScheme=normalizationScheme, polyOrder=polyOrder,
                    #generateMask
                    use_time_mask=use_time_mask, use_wave_mask=use_wave_mask,
                    plotMasks=plotMasks, maskRelativeCutoff=maskRelativeCutoff,
                    maskAbsoluteCutoff=maskAbsoluteCutoff,
                    maskSmoothingFactor=maskSmoothingFactor,
                    maskWindowSize=maskWindowSize,
                    unitPrefix=1000, verbose=superVerbose
  )

  allSysremData = hru.sysrem(obs.data, obs.error, nCycles=maxIterations, returnAll=True)

  detectionStrengths = []
  detectionCoords    = []
  iterativeSigMats   = []
  partialReport = partial(obs.reportOnSingleSysremIteration,
                          allSysremData=allSysremData,
                          kpRange=kpRange,
                          doVarianceWeight=doVarianceWeight,
                          # XCor Analysis KWs
                            # Generate XCM
                            normalizeXCM=normalizeXCM,
                            xcorMode=xcorMode,
                            # Generate SigMat
                            outputVelocities=outputVelocities,
                            # Normalize SigMat
                            rowByRow=rowByRow,
                            byPercentiles=byPercentiles,
                          # Report Detection Strength KWs
                            kpSearchExtent=kpSearchExtent,
                            vsysSearchExtent=vsysSearchExtent,
                            plotDetection=plotDetection,
                            savePrefix=savePrefix,
                            plotKpExtent=plotKpExtent,
                            plotVsysExtent=plotVsysExtent,
                            detectionUnitStr=detectionUnitStr,
                            showDetectionPlots=showDetectionPlots,
                            closeDetectionPlots=closeDetectionPlots,
                            detectionTitle=detectionTitle,
                            clim=clim,
                            cmap=cmap,
                            figsize=figsize,
                          # General
                            targetKp=targetKp,
                            targetVsys=targetVsys,
                            unitPrefix=unitPrefix,
                            verbose=superVerbose
                  )

  if cores == 1:
    seq = range(len(allSysremData))
    if verbose:
      seq = tqdm(seq, desc='Calculating Detection Strengths')

    for iteration in seq:
      detStrength, detCoords, sm = partialReport(iteration)

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)
      iterativeSigMats.append(sm)
  else:
    # Cores is not 1 -> want to use multiprocessing
    if verbose:
      pbar = tqdm(total=len(allSysremData), desc='Calculating Detection Strengths')

    pool = Pool(processes = cores)
    seq = pool.imap(partialReport, range(len(allSysremData)))

    # return seq
    for detVals in seq:
      detStrength, detCoords, sm = detVals

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)
      iterativeSigMats.append(sm)
      if verbose:
        pbar.update()

    if verbose:
      pbar.close()

  return detectionStrengths, detectionCoords
###