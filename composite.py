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
                  plotDetection=False, savePrefix=None,
                  doSaveData=False, returnSigMat=False,
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
                    kpSearchExtent=4, vsysSearchExtent=1,
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
                    injectedRelativeStrength=None,
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
    savePrefix = makePaths(savePrefix)
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

  if doSaveData:
    if savePrefix is None:
      raise valueError('savePrefix must be specified to save data.')
    else:
      saveDataName = savePrefix+'sysIt_'+str(iteration)

    saveDict = {}
    saveDict['sysremIterations'] = iteration
    injectedStr = f'10e{np.log10(injectedRelativeStrength):.1f}'
    saveData(theCopy.unNormedSigMat, saveDataName, kpRange, theCopy.crossCorVels,
             theCopy.planet, theCopy.instrument, theCopy.template, theCopy.date,
             theCopy.order, targetKp, targetVsys, saveDict,
             injectedStr=injectedStr, doPlot=False)

  if returnSigMat:
    return detStrength, detCoords, theCopy.unNormedSigMat, theCopy.crossCorVels

  else:
    return detStrength, detCoords

def reportSysremIterations(obs, kpRange,
                  maxIterations=10, cores=1,
                  plotDetection=False, savePrefix=None,
                  doSaveData=False, returnSigMats=False,
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
                    # LF Trends
                    doRemoveLFTrends=False,
                    nTrends=1,
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
                    kpSearchExtent=4, vsysSearchExtent=1,
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

  try:
    obs.wavelengths
  except AttributeError:
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
                    # LF Trends:
                    doRemoveLFTrends=doRemoveLFTrends, nTrends=nTrends,
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
  if returnSigMats:
    iterativeSigMats   = []
    crossCorVels       = []

  partialReport = partial(reportOnSingleSysremIteration,
                          obs=obs,
                          allSysremData=allSysremData,
                          kpRange=kpRange,
                          plotDetection=plotDetection,
                          savePrefix=savePrefix,
                          doSaveData=doSaveData,
                          returnSigMat=returnSigMats,
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
                            injectedRelativeStrength=injectedRelativeStrength,
                            unitPrefix=unitPrefix,
                            verbose=superVerbose
                  )

  if cores == 1:
    seq = range(len(allSysremData))
    if verbose:
      seq = tqdm(seq, desc='Calculating Detection Strengths')

    for iteration in seq:
      if returnSigMats:
        detStrength, detCoords, sigMat, ccvs = partialReport(iteration)

        iterativeSigMats.append(sigMat)
        crossCorVels.append(ccvs)

      else:
        detStrength, detCoords = partialReport(iteration)

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)

  else:
    # Cores is not 1 -> want to use multiprocessing
    if verbose:
      pbar = tqdm(total=len(allSysremData), desc='Calculating Detection Strengths')

    pool = Pool(processes = cores)
    seq = pool.imap(partialReport, range(len(allSysremData)))

    # return seq
    for detVals in seq:
      if returnSigMats:
        detStrength, detCoords, sigMat, ccvs = detVals

        iterativeSigMats.append(sigMat)
        crossCorVels.append(ccvs)
      else:
        detStrength, detCoords = detVals

      detectionStrengths.append(detStrength)
      detectionCoords.append(detCoords)
      
      if verbose:
        pbar.update()

    if verbose:
      pbar.close()

  if returnSigMats:
    return detectionStrengths, detectionCoords, iterativeSigMats, crossCorVels
  else:
    return detectionStrengths, detectionCoords
###

#-- Injection/Recovery
def injRec(planet, instruments, templates, dates, orders, tkp, tvs, kpr, exps,
           topSaveDir = 'plots/injRec/', normalizeXCM=True, **kwargs):
  '''
  '''
  outVels = [-220,220]
  injLocStr = str(tkp)+'_'+str(tvs)
  topSaveDir = topSaveDir+injLocStr+'/'
  makePaths(topSaveDir)

  # When only a single instrument is passed:
  if type(instruments) == str:
    instruments = [instruments]
    dates  = [dates]
    orders = [orders]

  pbarLength = np.sum(np.array(list(map(len,orders))) * list(map(len,dates)))
  pbarLength *= len(templates) * len(exps)

  pbar = tqdm(total=pbarLength, desc='Calculating')

  for tem in templates:

    for i, instrument in enumerate(instruments):

      for date in dates[i]:
        for order in orders[i]:

          savePrefix = topSaveDir+tem+'/'+date+'/order_'+str(order)+'/'
          makePaths(savePrefix)

          optItsPath = savePrefix+'opt/'
          makePaths(optItsPath)
          detLevels=[]
          optItsVals=[]

          obs = hrsObs('jsondb.json', planet, instrument, date, order, template=tem)
          obs.collectRawData()

          for exp in exps:
            inStrength = 10**exp
            inStrengthStr = '10e%.1f'%exp
            sysItsPath = savePrefix+inStrengthStr+'/'
            makePaths(sysItsPath)

            obsCopy = obs.copy()
            recStrengths, coords, sigMats, ccvs = reportSysremIterations(obsCopy,
              kpr,
              maxIterations=8,
              kpSearchExtent=5, vsysSearchExtent=1,
              doInjectSignal=True,
              targetKp=tkp, targetVsys=tvs,
              injectedRelativeStrength=inStrength,
              plotDetection=True,
              showDetectionPlots=True,
              closeDetectionPlots=True,
              outputVelocities=outVels,
              verbose=False,
              savePrefix=sysItsPath,
              doSaveData=True,
              returnSigMats=True,
              normalizeXCM=normalizeXCM,
              cores=10,
              **kwargs
            )

            # record the optimal
            sysIt = np.argmax(recStrengths)
            # copy the plot
            copyfile(sysItsPath+'sysIt_'+str(sysIt)+'.png', optItsPath+inStrengthStr+'.png')
            # copy the data
            copyfile(sysItsPath+'sysIt_'+str(sysIt)+'.pickle', optItsPath+inStrengthStr+'.pickle')
            # save the detection strength
            detLevels.append(recStrengths[sysIt])
            optItsVals.append(sysIt)

            pbar.update()

          # create detection levels plot
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

          # save detection levels data
          detLevelSave = {}
          detLevelSave['detLevels'] = detLevels
          detLevelSave['optIts'] = optItsVals
          detLevelSave['planet'] = planet
          detLevelSave['date'] = date
          detLevelSave['order'] = order
          detLevelSave['template'] = tem
          detLevelSave['exps'] = exps
          pickle.dump(detLevelSave,open(saveTrend+'order_'+str(order)+'.pickle','wb'))

  pbar.close()

def combineData(planet, instruments, templates, dates, orders, tkp, tvs, kpr,
                exps, topSaveDir='plots/injRec/', outX = np.arange(-200,200,1)
):
  injLocStr = str(tkp)+'_'+str(tvs)

  # When only a single instrument is passed:
  if type(instruments) == str:
    instruments = [instruments]
    dates  = [dates]
    orders = [orders]

  pbarLength = np.sum(np.array(list(map(len,orders))) * list(map(len,dates)))
  pbarLength *= len(templates) * len(exps)

  pbar = tqdm(total=pbarLength, desc='Calculating')

  for exp in exps:
    inStrength = 10**exp
    inStrengthStr = '10e%.1f'%exp

    for tem in templates:
      # SigMats, CCVs
      templateData = [[],[]]
      temPath = topSaveDir+injLocStr+'/'+tem+'/'

      combinedDatesPath = temPath+'combinedDates/'
      makePaths(combinedDatesPath)

      for i, instrument in enumerate(instruments):
        instrumentData = [[],[]]

        for date in dates[i]:
          datePath = temPath+date+'/'

          combinedOrdersPath = datePath+'combinedOrders/'
          makePaths(combinedOrdersPath)

          # SigMats, CCVs
          dateData = [[],[]]

          for order in orders[i]:
            dataPath = datePath+'order_'+str(order)+'/opt/'
            file = dataPath+inStrengthStr+'.pickle'

            data = pickle.load(open(file,'rb'))
            dateData[0].append(data['sigMat'])
            dateData[1].append(data['ccvs'])

            templateData[0].append(data['sigMat'])
            templateData[1].append(data['ccvs'])

            instrumentData[0].append(data['sigMat'])
            instrumentData[1].append(data['ccvs'])

            pbar.update()

          dateData = hru.addMatricies(*dateData, outX)
          dateSaveName = combinedOrdersPath+inStrengthStr

          saveData(dateData, dateSaveName, kpr, outX, planet, instrument,
            tem, date, orders[i], tkp, tvs, injectedStr=inStrengthStr)

        instrumentData = hru.addMatricies(*instrumentData, outX)
        instrumentSaveName = combinedDatesPath+inStrengthStr+'_'+instrument
        saveData(instrumentData, instrumentSaveName, kpr, outX, planet,
          instrument, tem, dates[i], orders[i], tkp, tvs, injectedStr=inStrengthStr)

      templateData = hru.addMatricies(*templateData, outX)
      temSaveName = combinedDatesPath+inStrengthStr
      saveData(templateData, temSaveName, kpr, outX, planet, instruments,
        tem, dates, orders, tkp, tvs, injectedStr=inStrengthStr)

  pbar.close()

def saveData(data, saveName, kpr, ccvs,
    planet, instrument, template, date, order,
    tkp, tvs, saveDict={}, injectedStr=None,
    doPlot=True, xlim=[-100,100]
):

  saveDict['sigMat'] = data
  saveDict['kpr'] = kpr
  saveDict['ccvs'] = ccvs

  saveDict['planet'] = planet
  saveDict['instrument'] = instrument
  saveDict['template'] = template
  saveDict['date'] = date
  saveDict['order'] = order
  saveDict['kp'] = tkp
  saveDict['vs'] = tvs

  title = planet+' '+template

  if injectedStr is not None:
    saveDict['injectedStr'] = injectedStr
    title+=' injected: '+injectedStr
  title+='\ndate: '+str(date)
  title+='\norder: '+str(order)

  pickle.dump(saveDict, open(saveName+'.pickle','wb'))

  if doPlot:
    hru.plotSigMat(hru.normalizeSigMat(data),ccvs,kpr,targetKp=tkp, targetVsys=tvs,
      xlim=xlim, title= title, saveName=saveName+'.png')