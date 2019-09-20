# todo input sanitization better
# todo if targetKp is none, use injected kp etc

import json
import copy
#import secrets

from hrsObs import *
from utility import *
import highResUtils as hru
import messenger

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import interpolate, signal
from shutil import copyfile

# from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
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
  doAutoTrimCols=True, plotTrim=False, colTrimFunc=hru.findEdgeCuts_xcor,
  neighborhood_size=30, gaussian_blur=10,
  edge=0, rightEdge=None, relative=True,
  #alignData
  alignmentIterations=1, alignmentPadLen=None,
  alignmentPeakHalfWidth=3, alignmentUpSampFactor=1000,
  #remove Trends:
  doRemoveLFTrends=False, nTrends=1,
  lfTrendMode=0,
  highPassFilter=False, hpKernel=65,
  #injectData
  doInjectSignal=False, injectedKp=None, injectedVsys=None,
  subtractSignal=False,
  removeNominal=False,
  injectedRelativeStrength=None, unitPrefix=1000,
  #normalize
  normalizationScheme='divide_row',polyOrder=2,
  #generateMask
  use_time_mask=True, use_wave_mask=False,
  plotMasks=False, maskRelativeCutoff=3,
  maskAbsoluteCutoff=0, maskSmoothingFactor=20,
  maskWindowSize=25, cela=12,
  # sysrem
  sysremIterations=None,
  stopBeforeSysrem=False,
  # Post Sysrem
  doVarianceWeight=True,
  verbose=False
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
    obs.removeLowFrequencyTrends(nTrends=nTrends, kernel=hpKernel,
      mode=lfTrendMode)

  if doInjectSignal:
    # print('---------------------------------')
    # print('----- Injecting Fake Signal -----')
    # print('---------------------------------')

    if removeNominal:
      obs.injectFakeSignal(injectedKp=obs.getNominalKp(), injectedVsys=obs.getNominalVsys(),
        relativeStrength=1, subtract=True, unitPrefix=unitPrefix)

    obs.injectFakeSignal(injectedKp=injectedKp, injectedVsys=injectedVsys,
                          relativeStrength=injectedRelativeStrength,
                          subtract=subtractSignal,
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

    if highPassFilter:
      print(f'hpfiltering {hpKernel}')
      trend = np.apply_along_axis(signal.medfilt, 1, obs.data, hpKernel)
      obs.data = obs.data - trend

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

#-- Helper Functions
def setObsParamFormat(templates, instruments, dates, orders):
  # handle multiple input types
  # When template is passed as a string:
  if type(templates) == str:
    templates = [templates]

  # When only a single instrument is passed:
  if type(instruments) == str:
    instruments = [instruments]

    if type(dates) == str:
      dates = [dates]

    if type(orders) == int:
      orders = [orders]

    dates = [dates]
    orders = [orders]

  else:
    if type(dates) == str:
      raise TypeError("Dates must be list of same length as instruments")
    if type(dates[0]) == str:
      dates = [[date] for date in dates]

    if type(orders) == int:
      raise TypeError("Orders must be list of same length as instruments")

    if type(orders[0]) == int:
      orders = [[order] for order in orders]

  return templates, instruments, dates, orders

def getExpectedOrbParams(dbName, planet):
  testObs = hrsObs(dbName, planet)
  ret = (testObs.orbParams['Kp'], testObs.orbParams['v_sys'])
  del testObs
  return ret

def setTargetParams(dbName, planet, targetKp, targetVsys):
  if targetKp is None or targetVsys is None:
    expectKp, expectVsys = getExpectedOrbParams(dbName, planet)
    if targetKp is None:
      targetKp = expectKp
    if targetVsys is None:
      targetVsys = expectVsys

  return targetKp, targetVsys

def setupInputs(dbName, planet, templates, instruments, dates, orders,
  topSaveDir, creatingData, injectedSignal,
  targetKp, targetVsys, injectionStrengths,
  subtractedInjection=False
):
  templates, instruments, dates, orders = setObsParamFormat(templates, instruments, dates, orders)

  pbarLength = np.sum(np.array(list(map(len,orders))) * list(map(len,dates)))
  pbarLength = pbarLength*len(templates)

  # If targetKp or targetVsys is none, set to expected values from db
  targetKp, targetVsys = setTargetParams(dbName, planet, targetKp, targetVsys)

  topDirs = {}

  if injectedSignal:
    injLocStr = str(targetKp)+'_'+str(targetVsys)
    for template in templates:
      if subtractedInjection:
        topDirs[template] = topSaveDir+'subtract_'+template+'/'+injLocStr+'/'
      else:
        topDirs[template] = topSaveDir+'add_'+template+'/'+injLocStr+'/'

    if isinstance(injectionStrengths, (float, int, str)):
      injectionStrengths = [injectionStrengths]

    if creatingData:
      # Assert that doInjectSignal and injectionStrengths match:
      if injectionStrengths is None:
        raise ValueError('Injection Strengths must be specified for doInjectSignal==True')
    else:
      # If not specified, use all available injectionStrengths
      if injectionStrengths is None:
        injectionStrengths=[]
        try:
          for f in os.listdir(topDirs[templates[0]]):
            if os.path.isdir(topDirs[templates[0]]+f):
              injectionStrengths.append(f)
        except FileNotFoundError as e:
          raise FileNotFoundError(f"Path {topDirs[templates[0]]} does not exist.")

    pbarLength = pbarLength * len(injectionStrengths)
  else:
    if injectionStrengths is not None:
      raise ValueError('Injection Strengths should be "None" for no injected signal')

    for template in templates:
      topDirs[template] = topSaveDir+'noInject/'
    injectionStrengths = [None]

  # If injectionStrengths is a string or float, put it in list form
  if not isinstance(injectionStrengths, (list, np.ndarray)):
    injectionStrengths = [injectionStrengths]

  return templates, instruments, dates, orders, targetKp, targetVsys, topDirs, injectionStrengths, pbarLength

def getInjectionStrengthString(injectionStrength, nDecimal=1, asPath=True):
  if injectionStrength is None:
    if asPath:
      return ''
    else:
      return 'None'

  if isinstance(injectionStrength, str):
    injectionStrengthString = injectionStrength
  else:
    injectionStrengthString = f'1e{np.log10(injectionStrength):.{nDecimal}f}'

  if asPath:
    return injectionStrengthString+'/'
  else:
    return injectionStrengthString

def getObsDataPath(template, date, order, injectionStrength=None, topDir=None):
  subPath = getInjectionStrengthString(injectionStrength)
  subPath += template+'/'+date+f'/order_{order}/'

  if topDir is None:
    return subPath

  if topDir[-1] != '/':
    topDir = topDir+'/'

  return topDir+subPath

def getTargetPath(targetKp, targetVsys, injectionStrength=None, topDir=None):
  subPath = getInjectionStrengthString(injectionStrength)
  subPath += f'target_{targetKp}_{targetVsys}'

  if topDir is None:
    return subPath
  else:
    return topDir+subPath

def getObsList(planet, templates, instruments, dates, orders,
  injectionStrengths=[None], topDirs=None
):
  obsList = []
  templates, instruments, dates, orders = setObsParamFormat(templates, instruments, dates, orders)

  for injectionStrength in injectionStrengths:
    for template in templates:
      for i, instrument in enumerate(instruments):
        for date in dates[i]:
          for order in orders[i]:
            try:
              topDir = topDirs[template]
            except (TypeError, KeyError):
              topDir = None

            obsData = {
              "planet": planet,
              "template": template,
              "instrument": instrument,
              "date": date,
              "order": order,
              "injectionStrength": injectionStrength,
              "path": getObsDataPath(template, date, order, injectionStrength, topDir)
            }
            obsList.append(obsData)
  return obsList

# todo
def saveSysremDictionaries(allSysremDicts,
  templates, topDirs,
  isSubtractedInjection=False,
  injectedKp=None, injectedVsys=None,
  targetKp=None, targetVsys=None,
  sysremSaveName=None, sysremComment=None,
  explicitSaveDir=None,

  creatingData=True, extraKeys=None,
):
  if injectedKp is None:
    injectedKp = targetKp
  if injectedVsys is None:
    injectedVsys = targetVsys

  for injectionTemplate in allSysremDicts.keys():

    for injectionStrength in allSysremDicts[injectionTemplate].keys():

      sysremDict = allSysremDicts[injectionTemplate][injectionStrength]
      sysremDict['targetKp'] = targetKp
      sysremDict['targetVsys'] = targetVsys
      sysremDict['comment'] = sysremComment

      if extraKeys is not None:
        for key, value in extraKeys.items():
          sysremDict[key] = value

      if injectionTemplate is None:
        sysremDict['injection'] = 'No Injection'

      else:
        if isSubtractedInjection:
          injectionString = 'Subtracted '+injectionTemplate+' '
        else:
          injectionString = 'Added '+injectionTemplate+' '
        injectionString = injectionString + getInjectionStrengthString(injectionStrength,asPath=False)
        sysremDict['injection'] = injectionString + f' at kp: {injectedKp}, vs: {injectedVsys}'

      if injectionTemplate is None:
        injectionTemplate = templates[0]

      if explicitSaveDir is None:
        saveDir = topDirs[injectionTemplate]+getInjectionStrengthString(injectionStrength, asPath=True)
        if not creatingData:
          if not os.path.isdir(saveDir):
            print(f"Warning: directory {saveDir} does not exist, no data written")
            continue
        saveDir = saveDir+f'target_{targetKp}_{targetVsys}/'
        makePaths(saveDir)
      else:
        saveDir = explicitSaveDir

      if sysremSaveName is None:
        sysremSaveName='sysrem'

      sysremFile = saveDir + sysremSaveName + '.pickle'
      if os.path.isfile(sysremFile):
        print('warning, sysremfile already exists and is overwritten')

      f = open(sysremFile, 'wb')
      pickle.dump(sysremDict, f)
      f.close()

def saveData(data, saveName, kpr, ccvs,
  planet, instrument, template, date, order,
  tkp, tvs, saveDict={}, injectedStr=None,
  doPlot=True, xlim=[-100,100],

  normalize=True,
  normalizeRowByRow=False,
  normalizeByPercentiles=False
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
  # title+='\norder: '+str(order)

  f = open(saveName+'.pickle','wb')
  pickle.dump(saveDict, f)
  f.close()

  if normalize:
    data = hru.normalizeSigMat(data,
        byPercentiles=normalizeByPercentiles, rowByRow=normalizeRowByRow)

  if doPlot:
    hru.plotSigMat(data,
      ccvs,kpr,targetKp=tkp, targetVsys=tvs,
      xlim=xlim, title= title, saveName=saveName+'.png')
###

#-- Single Operations
def generateSysremIterations(obs, kpRange,
  prepareFunction=None,
  maxIterations=10,
  saveDir=None,
  doOptimizeIterations=False,
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
    lfTrendMode=0,
    highPassFilter=False,
    hpKernel=65,
    #injectData
    doInjectSignal=False,
    injectedRelativeStrength=1/1000,
    doSubtractInjection=False,
    removeNominal=False,
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
  # Optimize Iterations:
    kpSearchExtent=5, vsysSearchExtent=1,
    plotOptIts=False, plotKpExtent=40, plotVsysExtent=50,
    saveOptItsPlotName=None, optItsTitle='',
    ba=True,
  # General
    targetKp=None, targetVsys=None,
    unitPrefix=1000, verbose=True
):
  superVerbose = (verbose>1)

  try:
    obs.wavelengths
  except AttributeError:
    obs.collectRawData()

  if prepareFunction is None:
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
      lfTrendMode=lfTrendMode,hpKernel=hpKernel,
      #injectData
      doInjectSignal=doInjectSignal,
      injectedKp=targetKp, injectedVsys=targetVsys,
      injectedRelativeStrength=injectedRelativeStrength,
      subtractSignal=doSubtractInjection,
      removeNominal=removeNominal,
      #normalize
      normalizationScheme=normalizationScheme, polyOrder=polyOrder,
      #generateMask
      use_time_mask=use_time_mask, use_wave_mask=use_wave_mask,
      plotMasks=plotMasks, maskRelativeCutoff=maskRelativeCutoff,
      maskAbsoluteCutoff=maskAbsoluteCutoff,
      maskSmoothingFactor=maskSmoothingFactor,
      maskWindowSize=maskWindowSize,
      unitPrefix=unitPrefix, verbose=superVerbose
    )
  else:
    prepareFunction(obs, doInjectSignal=doInjectSignal,
      injectedKp=targetKp, injectedVsys=targetVsys,
      injectedRelativeStrength=injectedRelativeStrength,
      subtractSignal=doSubtractInjection,
      normalizationScheme=normalizationScheme)

  allSysremData = hru.sysrem(obs.data, obs.error,
    nCycles=maxIterations, returnAll=True, verbose=superVerbose
  )

  analyzedData = []
  detectionStrengths = []

  seq = range(len(allSysremData))

  if verbose:
    seq = tqdm(seq, desc='Analyzing Sysrem Iterations')
  
  for iteration in seq:
    theCopy = obs.copy()
    theCopy.data = allSysremData[iteration]
    theCopy.log.append(f'Sysrem: {iteration} cycles')
    theCopy.sysremIterations = iteration

    if doVarianceWeight:
      theCopy.varianceWeight()

    if highPassFilter:
      trend = np.apply_along_axis(signal.medfilt, 1, theCopy.data, hpKernel)
      theCopy.data = theCopy.data - trend

    try:
      theCopy.applyMask()
    except AttributeError:
      pass

    xcorAnalysis(theCopy, kpRange,
     normalizeXCM=normalizeXCM,
     xcorMode=xcorMode,
     outputVelocities=outputVelocities,
     rowByRow=rowByRow,
     byPercentiles=byPercentiles,
     unitPrefix=unitPrefix,
     verbose=superVerbose
    )

    analyzedData.append(theCopy)

    if doOptimizeIterations:
      detStr, detCoords = theCopy.reportDetectionStrength(targetKp=targetKp, targetVsys=targetVsys,
        kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent,
        plotResult=plotOptIts, plotKpExtent=plotKpExtent, plotVsysExtent=plotVsysExtent,
        saveName=saveOptItsPlotName, title=optItsTitle)
      detectionStrengths.append(detStr)

    if saveDir is not None:
      fn = saveDir+f'sysIt_{iteration}.pickle'
      f = open(fn,'wb')
      pickle.dump(theCopy, f)
      f.close()

  if doOptimizeIterations:
    return analyzedData, detectionStrengths

  return analyzedData

def getDetectionStrengths(sysItsPath,
  targetKp=None, targetVsys=None,
  kpSearchExtent=5, vsysSearchExtent=1,
  filePrefix='sysIt_', fileSuffix='.pickle',
  useUnNormalized=False,
  useRowByRow=False,
  useByPercentiles=False,
  **kwargs
):
  detStrengthList = []

  # If sysItsPath is empty, return empty list
  try:
    fileList = os.listdir(sysItsPath)
  except FileNotFoundError:
    return []

  for file in fileList:
    thisPrefix = file[:len(filePrefix)]
    thisIteration = file[len(filePrefix):len(fileSuffix)]
    thisSuffix = file[len(fileSuffix):]

    if thisPrefix != filePrefix or thisSuffix != fileSuffix:
      continue

    f = open(sysItsPath+file,'rb')
    data = pickle.load(f)
    f.close()

    detStr, detCoords = data.reportDetectionStrength(targetKp=targetKp, targetVsys=targetVsys,
      kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent, 
      unNormedSigMat=useUnNormalized, rowByRow=useRowByRow, byPercentiles=useByPercentiles,
      **kwargs)

    detStrengthList.append(detStr)
  return detStrengthList

def analyzeWithNewTemplate(pathToData,
  newTemplate, kpRange, saveDir=None,
  filePrefix='sysIt_', fileSuffix='.pickle',

  doOptimizeIterations=True,
  targetKp=None, targetVsys=None,
  kpSearchExtent=5, vsysSearchExtent=1,

  normalizeXCM=True, xcorMode='same',
  # Generate SigMat
  outputVelocities=None,
  # Normalize SigMat
  doNormalizeSigMat=True,
  rowByRow=False, byPercentiles=False,
  # General
  unitPrefix=1000, verbose=False
):
  try:
    fileList = os.listdir(pathToData)
  except FileNotFoundError:
    raise FileNotFoundError

  detectionStrengths = []

  for file in fileList:
    thisPrefix = file[:len(filePrefix)]
    thisIteration = file[len(filePrefix):len(fileSuffix)]
    thisSuffix = file[len(fileSuffix):]

    if thisPrefix != filePrefix or thisSuffix != fileSuffix:
      continue

    f = open(pathToData+file,'rb')
    data = pickle.load(f)
    f.close()

    theCopy = data.copy()
    theCopy.template = newTemplate
    xcorAnalysis(theCopy, kpRange,
      normalizeXCM=normalizeXCM,
      xcorMode=xcorMode,
      outputVelocities=outputVelocities,
      doNormalizeSigMat=doNormalizeSigMat,
      rowByRow=rowByRow,
      byPercentiles=byPercentiles,
      unitPrefix=unitPrefix,
      verbose=verbose
    )

    if saveDir is not None:
      f = open(saveDir+file,'wb')
      pickle.dump(theCopy, f)
      f.close()

    if doOptimizeIterations:
      detStr, detCoords = theCopy.reportDetectionStrength(targetKp=targetKp, targetVsys=targetVsys,
        kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent)
      detectionStrengths.append(detStr)

  return detectionStrengths

def blah(obs, kpRange,
  sysremIterations=1,
  saveDir=None,
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
  # General
    targetKp=None, targetVsys=None,
    unitPrefix=1000, verbose=True
):
  superVerbose = (verbose>1)

  try:
    obs.wavelengths
  except AttributeError:
    obs.collectRawData()

  prepareData(obs,
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
    doInjectSignal=True,
    injectedKp=targetKp, injectedVsys=targetVsys,
    subtractSignal=True,
    injectedRelativeStrength=injectedRelativeStrength,
    #normalize
    normalizationScheme=normalizationScheme, polyOrder=polyOrder,
    #generateMask
    use_time_mask=use_time_mask, use_wave_mask=use_wave_mask,
    plotMasks=plotMasks, maskRelativeCutoff=maskRelativeCutoff,
    maskAbsoluteCutoff=maskAbsoluteCutoff,
    maskSmoothingFactor=maskSmoothingFactor,
    maskWindowSize=maskWindowSize,
    #sysrem
    sysremIterations=sysremIterations,
    stopBeforeSysrem=False,
    doVarianceWeight=doVarianceWeight,
    #general
    unitPrefix=unitPrefix, verbose=superVerbose
  )

  xcorAnalysis(obs, kpRange,
     normalizeXCM=normalizeXCM,
     xcorMode=xcorMode,
     outputVelocities=outputVelocities,
     rowByRow=rowByRow,
     byPercentiles=byPercentiles,
     unitPrefix=unitPrefix,
     verbose=superVerbose
  )

  if saveDir is not None:
    pickle.dump(obs, open(saveDir+f'sysIt_{sysremIterations}.pickle','wb'))

  return obs
###

#-- Multiprocess Friendly single operations:
def mpGenerateSysremIterations(i, obsList, kpRange, topDirs,
  dbName='jsondb.json', doRemoveLFTrends=None,
  **kwargs
):
  planet     = obsList[i]['planet']
  template   = obsList[i]['template']
  instrument = obsList[i]['instrument']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  injectionStrength = obsList[i]['injectionStrength']

  obs = hrsObs(dbName, planet, instrument, date, order, template=template)
  saveDir = getObsDataPath(template, date, order, injectionStrength, topDirs[template])
  makePaths(saveDir)

  fn = saveDir+f'sysIt_0.pickle'
  if os.path.isfile(fn):
    raise RuntimeError('Data in "'+saveDir+'" already exists!')

  if doRemoveLFTrends is None:
    if instrument == 'ishell':
      doRemoveLFTrends = True
    elif instrument == 'aries':
      doRemoveLFTrends = False
    else:
      raise ValueError(f'Instrument {instrument} is invalid')

  gsiOut = generateSysremIterations(obs, kpRange,
    saveDir=saveDir,
    injectedRelativeStrength=injectionStrength,
    doRemoveLFTrends=doRemoveLFTrends,
    **kwargs
  )

  return gsiOut, obsList[i]

def mpGetDetectionStrengths(i, obsList, topDirs, **kwargs):
  template   = obsList[i]['template']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  injectionStrength = obsList[i]['injectionStrength']

  saveDir = getObsDataPath(template, date, order, injectionStrength, topDirs[template])
  detStrengthList = getDetectionStrengths(saveDir, **kwargs)
  return detStrengthList, obsList[i]

def mpAnalyzeWithNewTemplate(i, obsList, kpRange, topDirs,
  **kwargs
):
  template   = obsList[i]['template']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  injectionStrength = obsList[i]['injectionStrength']

  newTemplate = obsList[i]['newTemplate']

  loadDataDir = getObsDataPath(template, date, order, injectionStrength, topDirs[template])
  saveDataDir = getObsDataPath(newTemplate, date, order, injectionStrength, topDirs[template])
  makePaths(saveDataDir)

  detStrengthList = analyzeWithNewTemplate(loadDataDir, newTemplate,
    kpRange, saveDir=saveDataDir, **kwargs)

  return detStrengthList, obsList[i]

def mpBlah(i, obsList, kpRange, **kwargs):
  obs = hrsObs(obsList[i]['dbName'], obsList[i]['planet'], obsList[i]['instrument'],
    obsList[i]['date'], obsList[i]['order'], template=obsList[i]['template'])

  sysremIterations = obsList[i]['sysremIterations']
  saveDir = obsList[i]['saveDir']
  doRemoveLFTrends = obsList[i]['doRemoveLFTrends']
  inStrength = obsList[i]['injectionStrength']

  blout = blah(obs, kpRange, sysremIterations=sysremIterations,
    saveDir=saveDir, doRemoveLFTrends=doRemoveLFTrends,
    injectedRelativeStrength=inStrength, **kwargs)

  return blout, obsList[i]
###


#-- Multi Obs Functions
def generateSysremLandscape(planet, templates, instruments, dates, orders,
  kpRange, topDir, cores=1, dbName='jsondb.json',

  doInjectSignal=False, injectionStrengths=None,
  targetKp=None, targetVsys=None,
  doSubtractInjection=False,

  doOptimizeIterations=True, excludeZeroIterations=True,
  kpSearchExtent=5, vsysSearchExtent=1,
  sysremSaveName=None, sysremComment=None,

  outputVelocities=[-500,500],
  maxIterations=8, normalizeXCM=True,
  doRemoveLFTrends=None,
  **kwargs
):
  newInputs = setupInputs(dbName, planet,
    templates, instruments, dates, orders,
    topDir, True, doInjectSignal,
    targetKp, targetVsys, injectionStrengths,
    subtractedInjection=doSubtractInjection)

  templates, instruments, dates, orders, targetKp, targetVsys, \
    topDirs, injectionStrengths, pbarLength = newInputs

  obsList = getObsList(planet, templates, instruments, dates, orders, injectionStrengths)

  if doOptimizeIterations:
    allDates = [item for sublist in dates for item in sublist]
    if doInjectSignal:
      injectionTemplates = templates
    else:
      injectionTemplates = [None]

    allSysremDicts = createNestedDict(injectionTemplates, injectionStrengths, templates, allDates)

  pbar = tqdm(total=pbarLength, desc='Calculating')

  partialFunc = partial(mpGenerateSysremIterations,
    obsList=obsList,
    kpRange=kpRange,
    topDirs=topDirs,
    doOptimizeIterations=doOptimizeIterations,
    kpSearchExtent=kpSearchExtent,
    vsysSearchExtent=vsysSearchExtent,
    maxIterations=maxIterations,
    doInjectSignal=doInjectSignal,
    targetKp=targetKp, targetVsys=targetVsys,
    doSubtractInjection=doSubtractInjection,
    normalizeXCM=normalizeXCM,
    outputVelocities=outputVelocities,
    doRemoveLFTrends=doRemoveLFTrends,
    dbName=dbName,
    verbose=False,
    **kwargs
  )

  if cores > 1:
    pool = mp.Pool(processes=cores)
    seq = pool.imap_unordered(partialFunc, range(len(obsList)))
  else:
    seq = []
    for i in range(len(obsList)):
      seq.append(partialFunc(i))
      pbar.update()

  for gsiOut, obsData in seq:

    if doOptimizeIterations:
      detStrengthList = gsiOut[1]

      if excludeZeroIterations:
        optIts = np.argmax(detStrengthList[1:])+1
      else:
        optIts = np.argmax(detStrengthList)

      if doInjectSignal:
        injectionTemplate = obsData['template']
      else:
        injectionTemplate = None

      allSysremDicts[injectionTemplate][obsData['injectionStrength']][obsData['template']][obsData['date']][obsData['order']] = optIts

    if cores>1:
      pbar.update()

  if doOptimizeIterations:
    saveSysremDictionaries(allSysremDicts,
      templates, topDirs,
      isSubtractedInjection=doSubtractInjection,
      targetKp=targetKp, targetVsys=targetVsys,
      sysremSaveName=sysremSaveName, sysremComment=sysremComment,
      extraKeys={'maxIterations': maxIterations},
      creatingData=True
    )

  pbar.close()  

def calculateOptimizedSysrem(planet, templates, instruments, dates, orders,
  topDir, dbName='jsondb.json',
  excludeZeroIterations=True,

  isInjectedSignal=False, injectionStrengths=None,
  isSubtractedInjection=False,
  injectedKp=None, injectedVsys=None,

  targetKp=None, targetVsys=None,
  kpSearchExtent=5, vsysSearchExtent=1,
  filePrefix='sysIt_', fileSuffix='.pickle',

  saveOutput=True,
  sysremComment=None, sysremSaveName=None,
  explicitSaveDir=None,

  useUnNormalized=False,
  useRowByRow=False,
  useByPercentiles=False,
  cores=1, **kwargs
):
  newInputs = setupInputs(dbName, planet,
    templates, instruments, dates, orders,
    topDir, False, isInjectedSignal,
    injectedKp, injectedVsys, injectionStrengths,
    subtractedInjection=isSubtractedInjection)

  templates, instruments, dates, orders, injectedKp, injectedVsys, \
    topDirs, injectionStrengths, pbarLength = newInputs

  obsList = getObsList(planet, templates, instruments, dates, orders, injectionStrengths)

  pbar = tqdm(total=pbarLength, desc='Calculating')

  targetKp, targetVsys = setTargetParams(dbName, planet, targetKp, targetVsys)

  allDates = [item for sublist in dates for item in sublist]
  if isInjectedSignal:
    injectionTemplates = templates
  else:
    injectionTemplates = [None]

  allSysremDicts = createNestedDict(injectionTemplates, injectionStrengths, templates, allDates)

  partialFunc = partial(mpGetDetectionStrengths,
    obsList=obsList,
    topDirs=topDirs,
    targetKp=targetKp, targetVsys=targetVsys,
    kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent,
    filePrefix=filePrefix, fileSuffix=fileSuffix,
    useUnNormalized=useUnNormalized,
    useRowByRow=useRowByRow,
    useByPercentiles=useByPercentiles,
    **kwargs
  )

  if cores > 1:
    pool = mp.Pool(processes=cores)
    seq = pool.imap_unordered(partialFunc, range(len(obsList)))
  else:
    seq = []
    for i in range(len(obsList)):
      seq.append(partialFunc(i))
      pbar.update()

  for detStrengthList, obsData in seq:

    if detStrengthList == []:
      pbar.update()
      continue

    if excludeZeroIterations:
      optIts = np.argmax(detStrengthList[1:])+1
    else:
      optIts = np.argmax(detStrengthList)

    if isInjectedSignal:
      injectionTemplate = obsData['template']
    else:
      injectionTemplate = None

    allSysremDicts[injectionTemplate][obsData['injectionStrength']][obsData['template']][obsData['date']][obsData['order']] = optIts

    if cores > 1:
      pbar.update()

  if saveOutput:
    saveSysremDictionaries(allSysremDicts,
      templates, topDirs,
      isSubtractedInjection=isSubtractedInjection,
      injectedKp=injectedKp, injectedVsys=injectedVsys,
      targetKp=targetKp, targetVsys=targetVsys,
      sysremSaveName=sysremSaveName, sysremComment=sysremComment,
      creatingData=False, explicitSaveDir=explicitSaveDir
      )
    pbar.close()

  else:
    pbar.close()
    return allSysremDicts

def applyNewTemplate(planet, newTemplates, instruments, dates, orders,
  kpRange, topDir, injectedTemplate,
  cores=1, dbName='jsondb.json',
  filePrefix='sysIt_', fileSuffix='.pickle',

  injectedSignal=False, injectionStrengths=None,
  targetKp=None, targetVsys=None,
  subtractInjection=False,

  doOptimizeIterations=True,
  kpSearchExtent=5, vsysSearchExtent=1,
  sysremSaveName=None, sysremComment=None,
  excludeZeroIterations=True,

  outputVelocities=[-500,500],
  normalizeXCM=True,
  **kwargs
):
  if isinstance(injectedTemplate, list) and len(injectedTemplate) != 1:
    raise ValueError('injectedTemplate must be a single template')

  newInputs = setupInputs(dbName, planet,
    injectedTemplate, instruments, dates, orders,
    topDir, False, injectedSignal,
    targetKp, targetVsys, injectionStrengths,
    subtractedInjection=subtractInjection)

  injectedTemplate, instruments, dates, orders, targetKp, targetVsys, \
    topDirs, injectionStrengths, pbarLength = newInputs
  iDontGetAmateurAstronomy=True

  if isinstance(newTemplates,str):
    newTemplates = [newTemplates]

  pbarLength = pbarLength * len(newTemplates)

  if doOptimizeIterations:
    allDates = [item for sublist in dates for item in sublist]
    allSysremDicts = createNestedDict(injectedTemplate, injectionStrengths, newTemplates, allDates)

  obsList = getObsList(planet, injectedTemplate, instruments, dates, orders, injectionStrengths)

  completeObsList = []
  for obs in obsList:
    for template in newTemplates:
      obsCopy = copy.deepcopy(obs)
      obsCopy['newTemplate'] = template
      completeObsList.append(obsCopy)

  partialFunc = partial(mpAnalyzeWithNewTemplate,
    obsList=completeObsList,
    kpRange=kpRange,
    topDirs=topDirs,
    filePrefix=filePrefix,
    fileSuffix=fileSuffix,
    doOptimizeIterations=doOptimizeIterations,
    targetKp=targetKp, targetVsys=targetVsys,
    kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent,
    normalizeXCM=normalizeXCM,
    outputVelocities=outputVelocities,
    **kwargs
  )

  pbar = tqdm(total=pbarLength, desc='Calculating')

  if cores > 1:
    pool = mp.Pool(processes=cores)
    seq = pool.imap_unordered(partialFunc, range(len(completeObsList)))
  else:
    seq = []
    for i in range(len(completeObsList)):
      seq.append(partialFunc(i))
      pbar.update()

  for detStrengthList, obsData in seq:
    if doOptimizeIterations:
      if excludeZeroIterations:
        optIts = np.argmax(detStrengthList[1:])+1
      else:
        optIts = np.argmax(detStrengthList)

      allSysremDicts[injectedTemplate[0]][obsData['injectionStrength']][obsData['template']][obsData['date']][obsData['order']] = optIts

    if cores>1:
      pbar.update()

  if doOptimizeIterations:
    saveSysremDictionaries(allSysremDicts,
      newTemplates, topDirs,
      isSubtractedInjection=subtractInjection,
      targetKp=targetKp, targetVsys=targetVsys,
      sysremSaveName=sysremSaveName, sysremComment=sysremComment,
      creatingData=True
    )
  pbar.close()

# todo sysremdict name function
# todo bigger sysremdicts
# todo optimize sysrem if need be
# directories as: add_temp/str/target__/sysrem.pick
# injectiontemplate
def combineData(planet, templates, instruments, dates, orders,
  topDir, explicitTopDir=False,
  sysremSaveName=None, explicitSysremFile=False,
  saveName='combined', explicitSavePath=False,
  sysremDict=None,
  saveString=None,

  injectedSignal=False, injectionStrengths=None,
  injectedKp=None, injectedVsys=None,
  subtractInjection=False,

  targetKp=None, targetVsys=None,
  filePrefix='sysIt_', fileSuffix='.pickle',
  saveDatesAndInstrs=False,
  outputVelocities=np.arange(-500,500),
  dbName='jsondb.json',

  normalize=True,
  normalizeByPercentiles=False,
  normalizeRowByRow=False,
  doPlot=True,
  **kwargs
):
  newInputs = setupInputs(dbName, planet,
    templates, instruments, dates, orders,
    topDir, False, injectedSignal,
    injectedKp, injectedVsys, injectionStrengths,
    subtractedInjection=subtractInjection)

  templates, instruments, dates, orders, injectedKp, injectedVsys, \
    topDirs, injectionStrengths, pbarLength = newInputs

  if targetKp is None:
    targetKp = injectedKp
  if targetVsys is None:
    targetVsys = injectedVsys

  targetKp, targetVsys = setTargetParams(dbName, planet, targetKp, targetVsys)

  if explicitTopDir:
    topDirs = dict((template, topDir) for template in templates)

  obsList = getObsList(planet, templates, instruments, dates, orders, injectionStrengths)

  pbar = tqdm(total=pbarLength, desc='Combining Data')

  for injectionStrength in injectionStrengths:
    for template in templates:
      saveDir = topDirs[template]
      saveDir += getInjectionStrengthString(injectionStrength, asPath=True)
      saveDir += f'target_{targetKp}_{targetVsys}/'

      if sysremDict is None:
        if sysremSaveName is None:
          thisSysremDict = saveDir + 'sysrem.pickle'
        else:
          if explicitSysremFile:
            thisSysremDict = sysremSaveName
          else:
            thisSysremDict = saveDir + sysremSaveName+'.pickle'

        f = open(thisSysremDict,'rb')
        thisSysremDict = pickle.load(f)
        f.close()

      else:
        thisSysremDict = sysremDict

      if explicitSavePath:
        saveDir = saveName
      else:
        saveDir = saveDir+saveName+'/'
        makePaths(saveDir)

      templateData = [[],[]]

      for i, instrument in enumerate(instruments):
        instrumentData = [[],[]]

        for date in dates[i]:
          dateData = [[],[]]
          for order in orders[i]:
            dataDir = getObsDataPath(template, date, order, injectionStrength, topDirs[template])
            sysremIterations = thisSysremDict[template][date][order]
            dataFile = filePrefix+str(sysremIterations)+fileSuffix

            f = open(dataDir+dataFile,'rb')
            obs = pickle.load(f)
            f.close()

            dateData[0].append(obs.unNormedSigMat)
            dateData[1].append(obs.crossCorVels)

            templateData[0].append(obs.unNormedSigMat)
            templateData[1].append(obs.crossCorVels)

            instrumentData[0].append(obs.unNormedSigMat)
            instrumentData[1].append(obs.crossCorVels)

            pbar.update()

          try:
            injectedStr = thisSysremDict['injection']
          except KeyError:
            injectedStr = None

          if injectionStrength is None:
            injectedStr = None

          if saveDatesAndInstrs:
            dateData = hru.addMatricies(*dateData, outputVelocities)

            dateSavePath = saveDir+'dates/'
            makePaths(dateSavePath)
            dateSaveName = dateSavePath+date+'_'+template

            saveData(dateData, dateSaveName, obs.kpRange, outputVelocities,
              planet, instrument, template, date, orders[i],
              targetKp, targetVsys, injectedStr=injectedStr,
              saveDict={"sysrem": thisSysremDict},
              normalize=normalize,
              normalizeRowByRow=normalizeRowByRow,
              normalizeByPercentiles=normalizeByPercentiles,
              doPlot=doPlot)

        if saveDatesAndInstrs:
          instrumentData = hru.addMatricies(*instrumentData, outputVelocities)

          instSavePath = saveDir+'instruments/'
          makePaths(instSavePath)
          instrumentSaveName = instSavePath+instrument+'_'+template

          saveData(instrumentData, instrumentSaveName, obs.kpRange, outputVelocities,
            planet, instrument, template, dates[i], orders[i],
            targetKp, targetVsys, injectedStr=injectedStr,
            saveDict={"sysrem": thisSysremDict},
            normalize=normalize,
            normalizeRowByRow=normalizeRowByRow,
            normalizeByPercentiles=normalizeByPercentiles,
            doPlot=doPlot)

      templateData = hru.addMatricies(*templateData, outputVelocities)

      if saveString is None:
        temSaveString=template
      else:
        temSaveString = saveString
      templateSaveName = saveDir+temSaveString

      saveData(templateData, templateSaveName, obs.kpRange, outputVelocities,
          planet, instruments, template, dates, orders,
          targetKp, targetVsys, injectedStr=injectedStr,
          saveDict={"sysrem": thisSysremDict},
          normalize=normalize,
          normalizeRowByRow=normalizeRowByRow,
          normalizeByPercentiles=normalizeByPercentiles,
          doPlot=doPlot)

  pbar.close()



  # print(allSysremDicts)

def makeLargeXCM(planet, templates, instruments, dates, orders,
  kpRange, topDir, cores=1, dbName='jsondb.json',
  isInjectedSignal=False, injectedKp=None, injectedVsys=None,
  injectionStrengths=None,subtractInjection=False,

  outputVelocities=np.arange(-500,500,1),
  **kwargs
):

  newInputs = setupInputs(dbName, planet,
    templates, instruments, dates, orders,
    topDir, False, isInjectedSignal,
    injectedKp, injectedVsys, injectionStrengths,
    subtractedInjection=subtractInjection)

  templates, instruments, dates, orders, injectedKp, injectedVsys, \
    topDirs, injectionStrengths, pbarLength = newInputs

  allLargeXCMs = {}
  for injectionStrength in injectionStrengths:
    injectionStrengthString = getInjectionStrengthString(injectionStrength,
      asPath=False)
    allLargeXCMs[injectionStrengthString] = {}

    for template in templates:
      saveDir = topDirs[template]
      saveDir += getInjectionStrengthString(injectionStrength, asPath=True)
      saveDir += f'target_{injectedKp}_{injectedVsys}/'

      thisSysremDict = saveDir + 'sysrem.pickle'

      f = open(thisSysremDict,'rb')
      thisSysremDict = pickle.load(f)
      f.close()

      allDateData = []

      for i, instrument in enumerate(instruments):
        instrumentData = [[],[]]

        for date in dates[i]:
          dateData = [[],[],0]
          for order in orders[i]:
            dataDir = getObsDataPath(template, date, order, injectionStrength, topDirs[template])
            sysremIterations = thisSysremDict[template][date][order]
            dataFile = 'sysIt_'+str(sysremIterations)+'.pickle'

            f = open(dataDir+dataFile,'rb')
            obs = pickle.load(f)
            f.close()

            dateData[0].append(obs.unTrimmedXCM)
            dateData[1].append(obs.unTrimmedXCV)
            dateData[2] = obs.getPhases()

          allDateData.append(dateData)

      largeXCM = []
      allPhases = []
      for i in range(len(allDateData)):
        xcms, ccvs, phases = allDateData[i]
        dateXCM = hru.addMatricies(xcms, ccvs, outputVelocities)
        for j in range(len(phases)):
          largeXCM.append(dateXCM[j])
          allPhases.append(phases[j])

      sortedArgs = np.argsort(allPhases)
      allPhases = np.array(allPhases)[sortedArgs]
      largeXCM  = np.array(largeXCM)[sortedArgs]
      allLargeXCMs[injectionStrengthString][template] = {
        'xcm': largeXCM,
        'phases': allPhases,
        'ccvs': outputVelocities
      }

  return collapseDict(allLargeXCMs)


def hi(planet, templates, instruments, dates, orders,
  topSaveDir, kpRange, outputVelocities = np.arange(-500,500),
  injectionStrengths=None,
  targetKp=None, targetVsys=None,
  sysremFile='sysrem',
  dbName='jsondb.json', doRemoveLFTrends=None,
  cores=1, **kwargs
):
  newInputs = setupInputs(dbName, planet,
    templates, instruments, dates, orders,
    topSaveDir, True, True,
    targetKp, targetVsys, injectionStrengths)

  templates, instruments, dates, orders, targetKp, targetVsys, \
    _, injectionStrengths, pbarLength = newInputs

  topSaveDir = topSaveDir+f'noInject/target_{targetKp}_{targetVsys}/'
  saveDataDir = topSaveDir+'subtract/'

  sysremDictLoc = topSaveDir+sysremFile+'.pickle'
  sysremDict = pickle.load(open(sysremDictLoc,'rb'))

  pbar = tqdm(total=pbarLength, desc='Calculating')

  obsList=[]

  for inStrength in injectionStrengths:
    inStrengthStr = f'10e{np.log10(inStrength):.1f}/'
    injTopDir = saveDataDir+inStrengthStr

    makePaths(injTopDir)
    pickle.dump(sysremDict, open(injTopDir+sysremFile+'.pickle','wb'))

    for template in templates:

      for i, instrument in enumerate(instruments):
        # set doRemoveLFTrends by instrument if not specified
        if doRemoveLFTrends is None:
          if instrument == 'aries':
            doRemoveLFTrends = False
          elif instrument == 'ishell':
            doRemoveLFTrends = True
          else:
            raise ValueError(f'Instrument {instrument} is invalid')

        for date in dates[i]:

          for order in orders[i]:

            saveDir = injTopDir+template+'/'+date+f'/order_{order}/'
            makePaths(saveDir)

            obsList.append(
              {
                "dbName": dbName,
                "planet": planet,
                "instrument": instrument,
                "date": date,
                "order": order,
                "template": template,
                "injectionStrength": inStrength,
                "saveDir": saveDir,
                "doRemoveLFTrends": doRemoveLFTrends,
                "sysremIterations": sysremDict[template][date][order]
              }
            )
  
  pool = mp.Pool(processes=cores)
  partialFunc = partial(mpBlah,
    obsList=obsList,
    kpRange=kpRange,
    targetKp=targetKp, targetVsys=targetVsys,
    outputVelocities=outputVelocities,
    verbose=False,
    **kwargs
  )

  seq = pool.imap_unordered(partialFunc, range(len(obsList)))

  for blout, obsData in seq:
    pbar.update()

  pbar.close()

  combineData(planet, templates, instruments, dates, orders, saveDataDir,
    useTargetDirectory=False,forceTopSaveDir=True,
    injectionStrengths=injectionStrengths, isInjectedSignal=True,
    outputVelocities=outputVelocities)

  combineDir = saveDataDir+'/combined/'
  makePaths(combineDir)

  for inStrength in injectionStrengths:
    inStrengthStr = f'10e{np.log10(inStrength):.1f}/'
    injTopDir = saveDataDir+inStrengthStr

    for template in templates:
      templateDir = combineDir+template+'/'
      makePaths(templateDir)

      copyfile(injTopDir+f'/combined/{template}.png', templateDir+inStrengthStr[:-1]+'.png') 
###

def falsePositiveTest(planet, templates, instruments, dates, orders, topSaveDir,
  kpRange, targetVsysRange, messageAtEnd=False, verbose=False,
  diradd = 'fptest/',
  kpSearchExtent=5, vsysSearchExtent=1, outputVelocities=np.arange(-500,500)
):
  fpDir = topSaveDir+'noInject/'+diradd
  sysremDir = fpDir+'sysrem/'
  makePaths(sysremDir)

  if verbose:
    seq = tqdm(targetVsysRange, desc='Vsys Range: ')
  else:
    seq = targetVsysRange
  for targetVsys in seq:
    fileSaveName = f'tv_{targetVsys}'

    calculateOptimizedSysrem(planet, templates, instruments, dates, orders,
      topSaveDir, targetVsys=targetVsys, sysremSaveName=fileSaveName,
      explicitSaveDir=sysremDir, kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent)

    combineData(planet, templates, instruments, dates, orders, topSaveDir,
      targetVsys=targetVsys, doPlot=False, saveName=fpDir,
      explicitSavePath=True, saveString=fileSaveName,
      sysremSaveName=sysremDir+fileSaveName+'.pickle',
      explicitSysremFile=True, outputVelocities=outputVelocities)

  if messageAtEnd:
    messenger.sms('Done with False Positive test.')

def analyzeFalsePositiveTest(planet, templates, instruments, dates, orders, topSaveDir, normalize=True,
  normalizeRowByRow=True, normalizeByPercentiles=True,
  dbName='jsondb.json', diradd='fptest/',
  kpSearchExtent=10, vsysSearchExtent=1,
  makeMovie=False
):
  fileDir = topSaveDir+'noInject/'+diradd

  reportedStrengths = []

  files = os.listdir(fileDir)
  tvRange = []
  for f in files:
    try:
      n = int(f.split('_')[1].split('.')[0])
      tvRange.append(n)
    except IndexError:
      pass

  tvRange.sort()
  for tv in tvRange:
    fn = fileDir+f'tv_{tv}.pickle'
    with open(fn, 'rb') as f:
      data = pickle.load(f)

    sigMat = data['sigMat']
    ccvs = data['ccvs']
    kpr = data['kpr']

    if normalize:
      normalizedSigMat = hru.normalizeSigMat(sigMat, 
        rowByRow=normalizeRowByRow, byPercentiles=normalizeByPercentiles)
    else:
      normalizedSigMat = sigMat
    report = hru.reportDetectionStrength(normalizedSigMat, ccvs, kpr,
      targetKp=getExpectedOrbParams(dbName, planet)[0],
      targetVsys=tv,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent)
    reportedStrengths.append(report[0])

  if makeMovie:
    print(1)
  return tvRange, reportedStrengths