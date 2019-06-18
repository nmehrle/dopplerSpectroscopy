import json

from hrsObs import *
from utility import *
import highResUtils as hru

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import interpolate
from shutil import copyfile


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

#-- Default
def prepareData(obs,
                  # TrimData
                  doAutoTrimCols=True, plotTrim=False,
                  colTrimFunc=hru.findEdgeCuts_xcor,
                  neighborhood_size=30, gaussian_blur=10,
                  edge=0, rightEdge=None, relative=True,
                  #alignData
                  alignmentIterations=1, alignmentPadLen=None,
                  alignmentPeakHalfWidth=3, alignmentUpSampFactor=1000,
                  #remove Trends:
                  doRemoveLFTrends=False, nTrends=1,
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

  if doRemoveLFTrends:
    # After generate Mask?
    obs.removeLowFrequencyTrends(nTrends=nTrends)

  if doInjectSignal:
    # print('---------------------------------')
    # print('----- Injecting Fake Signal -----')
    # print('---------------------------------')

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

  if doNormalizeSigMat:
    obs.reNormalizeSigMat(rowByRow=rowByRow, byPercentiles=byPercentiles)
###

#-- Sysrem Optimizing
def reportOnSingleSysremIteration(iteration, obs, allSysremData, kpRange,
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

  xcorAnalysis(theCopy, kpRange,
                       normalizeXCM=normalizeXCM,
                       xcorMode=xcorMode,
                       outputVelocities=outputVelocities,
                       rowByRow=rowByRow,
                       byPercentiles=byPercentiles,
                       unitPrefix=unitPrefix,
                       verbose=verbose
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

  return detStrength, detCoords, theCopy.unNormedSigMat, theCopy.crossCorVels

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
                    plotKpExtent=50, plotVsysExtent=80,
                    detectionUnitStr='km/s',
                    showDetectionPlots=True,
                    closeDetectionPlots=False,
                    detectionTitle='',
                    clim=[None,None],
                    cmap='viridis',
                    figsize=None,
                  # General
                    targetKp=None, targetVsys=None,
                    unitPrefix=1000, verbose=True
):
  '''
    Given a hrsObs object, performs the data preparation and analysis,
    and then reports the detection strength for each sysremIteration up to the
    maxIterations specified.

    Can be done in multiprocessing if cores > 1
  '''

  superVerbose = (verbose>1)

  obs.collectRawData()
  prepareData(obs, stopBeforeSysrem=True,
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

  if verbose:
    print('Running Sysrem')
  allSysremData = hru.sysrem(obs.data, obs.error, nCycles=maxIterations, returnAll=True)

  if verbose:
    print('Done with Sysrem')

  detectionStrengths = []
  detectionCoords    = []
  iterativeSigMats   = []
  crossCorVels       = []
  partialReport = partial(reportOnSingleSysremIteration,
                          obs=obs,
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
      detStrength, detCoords, sigMat, ccvs = partialReport(iteration)

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)
      iterativeSigMats.append(sigMat)
      crossCorVels.append(ccvs)
  else:
    # Cores is not 1 -> want to use multiprocessing
    if verbose:
      pbar = tqdm(total=len(allSysremData), desc='Calculating Detection Strengths')

    pool = Pool(processes = cores)
    seq = pool.imap(partialReport, range(len(allSysremData)))

    # return seq
    for detVals in seq:
      detStrength, detCoords, sigMat, ccvs = detVals

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)
      iterativeSigMats.append(sigMat)
      crossCorVels.append(ccvs)
      if verbose:
        pbar.update()

    if verbose:
      pbar.close()

  return detectionStrengths, detectionCoords, iterativeSigMats, crossCorVels
###

#-- Injection/Recovery
def injRec(planet, instrument, templates, dates, orders, tkp, tvs, kpr, exps,
           topSaveDir = 'tests/injRec/'):
  '''
  '''
  outVels = [-200,200]
  for tem in tqdm(templates,desc='templates'):
    for date in tqdm(dates,desc=tem+': dates'):
      for order in tqdm(orders, desc=tem+' '+date+': orders'):

        injLocStr = str(tkp)+'_'+str(tvs)
        savePrefix = topSaveDir+injLocStr+'/'+tem+'/'+date+'/order_'+str(order)+'/'
        makePaths(savePrefix)

        optItsPath = savePrefix+'opt/'
        makePaths(optItsPath)
        detLevels=[]

        for exp in tqdm(exps, desc=tem+' '+date+' '+str(order)+': strengths'):
          inStrength = 10**exp
          inStrengthStr = '10e%.1f'%exp
          sysItsPath = savePrefix+inStrengthStr+'/'
          makePaths(sysItsPath)

          obs = hrsObs('jsondb.json', planet, instrument, date, order, template=tem)
          recStrengths, coords, sigMats, ccvs = reportSysremIterations(obs, kpr, 
                                                  maxIterations=8,
                                                  doInjectSignal=True,
                                                  targetKp=tkp, targetVsys=tvs,
                                                  injectedRelativeStrength=inStrength,
                                                  plotDetection=True,
                                                  showDetectionPlots=True,
                                                  closeDetectionPlots=True,
                                                  outputVelocities=outVels,
                                                  verbose=False,
                                                  savePrefix=sysItsPath,
                                                  cores=10
                                                )

          # record the optimal
          sysIt = np.argmax(recStrengths)
          # copy the plot
          copyfile(sysItsPath+'sysIt_'+str(sysIt)+'.png', optItsPath+inStrengthStr+'.png')
          # save the detection strength
          detLevels.append(recStrengths[sysIt])

        plt.figure()
        plt.plot(detLevels)
        xtickind = 3
        plt.xticks(np.arange(len(detLevels))[::xtickind], np.round(exps[::xtickind],1))

        plt.title(planet + ' -- '+date+' -- order: '+str(order)+'\n '+tem+' injection/recovery')
        plt.ylabel('Recovered Signal Level')
        plt.xlabel('Injected Signal Strength (log)')

        saveTrend = '/'.join(savePrefix.split('/')[:-2])+'/trends/'
        makePaths(saveTrend)

        plt.savefig(saveTrend+'order_'+str(order)+'.png')
        plt.show()

# def dateInjRec(planet, instrument, tem, dates, orders, tkp, tvs, kpr, exps,
               # sysIts=[]):
  # print(1)
###