import numpy as np
import copy
import multiprocessing as mp
import seaborn as sns
import matplotlib.pyplot as plt

from functools import partial
from scipy.stats import norm
# import secrets

from hrsObs import *
from utility import *
import highResUtils as hru

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


class Collection:

  # todo init (upsand, inverted) sets the rest of it
  def __init__(self, planet, template, instruments, dates, orders,
    rootDir=None, explicitRootDir=False,
    dbName='jsondb.json',
    targetKp=None, targetVsys=None
  ):
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

    self.planet      = planet
    self.template    = template
    self.instruments = instruments
    self.dates       = dates
    self.orders      = orders
    self.dbName = dbName

    self.pbarLength = np.sum(np.array(list(map(len,orders))) * list(map(len,dates)))

    # Unless an injection is specified, assume noInject
    self.injectedSignal = False
    self.injectionStrength = None
    self.injectionKp = None
    self.injectionVsys = None
    self.injectionTemplate = None

    if targetKp is None:
      self.autoSetTargetKp = True
    else:
      self.autoSetTargetKp = False
      
    if targetVsys is None:
      self.autoSetTargetVsys = True
    else:
      self.autoSetTargetVsys = False

    self._targetKp = targetKp
    self._targetVsys = targetVsys

    self.autoSetTargetParams()

    if rootDir is None:
      self.rootDir = 'output/'+planet+'/'
    else:
      self.rootDir = rootDir

    self.topDir = self.rootDir + 'noInject/'
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

    self.setObsList()

  @property
  def targetKp(self):
    return self._targetKp
  
  @targetKp.setter
  def targetKp(self, value):
    if value is None:
      self.autoSetTargetKp = True
      self.autoSetTargetParams()

    else:
      self.autoSetTargetKp = False
      self._targetKp = value

    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

  @property
  def targetVsys(self):
    return self._targetVsys
  
  @targetVsys.setter
  def targetVsys(self, value):
    if value is None:
      self.autoSetTargetVsys = True
      self.autoSetTargetParams()

    else:
      self.autoSetTargetVsys = False
      self._targetVsys = value

    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

  def autoSetTargetParams(self):
    needsExpectedKp = self.autoSetTargetKp and (self.injectionKp is None)
    needsExpectedVsys = self.autoSetTargetVsys and (self.injectionVsys is None)

    if needsExpectedKp or needsExpectedVsys:
      expectKp, expectVsys = self.getExpectedOrbParams()

    if self.autoSetTargetKp:
      if self.injectionKp is None:
        self._targetKp = expectKp
      else:
        self._targetKp = self.injectionKp

    if self.autoSetTargetVsys:
      if self.injectionVsys is None:
        self._targetVsys = expectVsys
      else:
        self._targetVsys = self.injectionVsys

  def setInjection(self, injectionKp, injectionVsys, injectionStrength=1):
    self.injectedSignal = True
    self.injectionKp = injectionKp
    self.injectionVsys = injectionVsys
    self.injectionTemplate = self.template
    self.injectionStrength = injectionStrength

    if self.autoSetTargetKp:
      self._targetKp = injectionKp

    if self.autoSetTargetVsys:
      self._targetVsys = injectionVsys

    topDir = self.rootDir + f'inject_{self.template}/'+str(injectionKp)+'_'+str(injectionVsys)+'/'
    topDir = topDir + getInjectionString(injectionStrength, asPath=True)
    self.topDir = topDir
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

    self.setObsList()

  def clearInjection(self):
    self.injectedSignal = False
    self.injectionStrength = None
    self.injectionTemplate = None
    self.injectionKp = None
    self.injectionVsys = None

    self.autoSetTargetParams()

    self.topDir = self.rootDir + 'noInject/'
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

    self.setObsList()

  def setObsList(self):
    obsList = []

    for i, instrument in enumerate(self.instruments):
      for date in self.dates[i]:
        for order in self.orders[i]:
          path = getObsDataPath(self.template, date, order, self.topDir)
          obsData = {
            "planet": self.planet,
            "template": self.template,
            "instrument": instrument,
            "date": date,
            "order": order,
            "injectionStrength": self.injectionStrength,
            "topDir": self.topDir,
            "path": path
          }

          obsList.append(obsData)

    self.obsList = obsList

  def setTarget(self, targetKp=None, targetVsys=None):
    if targetKp is not None:
      self.targetKp = targetKp

    if targetVsys is not None:
      self.targetVsys = targetVsys

    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

  def getTarget(self):
    return self.targetKp, self.targetVsys

  def getExpectedOrbParams(self):
    testObs = hrsObs(self.dbName, self.planet)
    ret = (testObs.orbParams['Kp'], testObs.orbParams['v_sys'])
    del testObs
    return ret

  def getInjectionStrengths(self):
    injectionStrengths = []
    try:
      for subPath in os.listdir(self.topDir):
        if os.path.isdir(self.topDir+subPath):
          injectionStrengths.append(subPath)
    except FileNotFoundError as e:
      raise FileNotFoundError(f"Path {topDir} does not exist, no Injection Strengths found")

    if injectionStrengths == []:
      raise ValueError(f'No injectionStrengths found for {topDir}.')

    return injectionStrengths

  def getSysremParams(self, saveName='sysrem'):
    fn = self.targetPath+saveName+'.pickle'
    with open(fn,'rb') as f:
      return pickle.load(f)

  def saveSysrem(self, sysremDict,
    comment=None, extraKeys=None,
    saveName='sysrem', explicitSaveName=False,
    targetKp=None, targetVsys=None
  ):
    if targetKp is None:
      targetKp = self.targetKp
    if targetVsys is None:
      targetVsys = self.targetVsys

    if not explicitSaveName:
      savePath = getTargetPath(targetKp, targetVsys, self.topDir)
      makePaths(savePath)
      saveName = savePath+f"{saveName}.pickle"

    sysremDict['collection'] = self

    sysremDict['planet'] = self.planet

    sysremDict['targetKp'] = targetKp
    sysremDict['targetVsys'] = targetVsys

    sysremDict['injectedSignal'] = self.injectedSignal
    sysremDict['injectionKp'] = self.injectionKp
    sysremDict['injectionVsys'] = self.injectionVsys
    sysremDict['injectionStrength'] = self.injectionStrength
    sysremDict['injectionTemplate'] = self.injectionTemplate

    sysremDict['comment'] = comment

    if extraKeys is not None:
      for key, value in extraKeys.items():
        sysremDict[key] = value

    if os.path.exists(saveName):
      with open(saveName,'rb') as f:
        existantSysremDict = pickle.load(f)

      # Merge the two dictionaries
      # Assert all common keys are equal except templates, comment, collection
      # Copy over all unshared keys
      # Overwrite old optimal sysrem iteration values
      for key in existantSysremDict.keys():
        if key in sysremDict.keys():
          try:
            # If keys are the same, dont worry
            assert existantSysremDict[key] == sysremDict[key], \
              f"New and Old Sysrem Dictionaries at {saveName} are different. Key {key} doesn't match"
          except Exception as e:
            # Shared Keys
            if key == 'comment':
              # Merge Comments:
              sysremDict['comment'] = [existantSysremDict[key], comment]
            elif key == self.template:
              # Merge Template:
              for dateKey in existantSysremDict[key]:
                if dateKey in sysremDict[key]:
                  existantSysremDict[key][dateKey].update(sysremDict[key][dateKey])
                  sysremDict[key][dateKey] = existantSysremDict[key][dateKey]
                else:
                  sysremDict[key][dateKey] = existantSysremDict[key][dateKey]
            elif key == 'collection':
              sysremDict['collection'] = 'Merged'
            else:
              raise(e)
        else:
          # Unshared keys
          # Copy from old dict to new one
          sysremDict[key] = existantSysremDict[key]

    with open(saveName,'wb') as f:
      pickle.dump(sysremDict, f)

  def constantSysrem(self, value):
    print(1)

  def saveCombinedData(self, data, crossCorVels, kpRange,
    saveName, saveDict={},

    targetKp=None, targetVsys=None,
    instrument=None, date=None, order=None,

    doPlotSigMat=True, xlim=[-100,100],
    nDecimal=2,

    normalize=True,
    normalizeRowByRow=True,
    normalizeByPercentiles=True
  ):
    if targetKp is None:
      targetKp = self.targetKp

    if targetVsys is None:
      targetVsys = self.targetVsys

    saveDict['sigMat'] = data
    saveDict['crossCorVels'] = crossCorVels
    saveDict['kpRange'] = kpRange

    saveDict['planet'] = self.planet
    saveDict['template'] = self.template

    saveDict['targetKp'] = targetKp
    saveDict['targetVsys'] = targetVsys

    saveDict['injectedSignal'] = self.injectedSignal
    saveDict['injectionKp'] = self.injectionKp
    saveDict['injectionVsys'] = self.injectionVsys
    saveDict['injectionStrength'] = self.injectionStrength

    if instrument is None:
      instrument = self.instruments
    saveDict['instrument'] = instrument

    if date is None:
      date = self.dates
    saveDict['date'] = date

    if order is None:
      order = self.orders
    saveDict['order'] = order

    title = self.planet+' '+self.template
    title += '\nInjection: '+getInjectionString(self.injectionStrength, asPath=False)
    saveDict['title'] = title

    with open(saveName+'.pickle','wb') as f:
      pickle.dump(saveDict, f)

    if doPlotSigMat:
      if normalize:
        data = hru.normalizeSigMat(data,
          byPercentiles=normalizeByPercentiles,
          rowByRow=normalizeRowByRow)

      hru.plotSigMat(data, crossCorVels, kpRange,
        targetKp=targetKp, targetVsys=targetVsys,
        title=title, xlim=xlim,
        nDecimal=nDecimal,
        saveName=saveName+'.png')

  #-- Main
  def generateSysremLandscape(self,
    kpRange, cores=1,

    doOptimizeIterations=True, excludeZeroIterations=True,
    kpSearchExtent=5, vsysSearchExtent=1,
    sysremSaveName='sysrem', sysremComment=None,

    outputVelocities=[-500,500],
    maxIterations=8, normalizeXCM=True,
    doRemoveLFTrends=None,
    overwrite=False,
    **kwargs
  ):
    obsList = self.obsList

    if doOptimizeIterations:
      allDates = [item for sublist in self.dates for item in sublist]

      sysremDict = createNestedDict([self.template], allDates)

    pbar = tqdm(total=self.pbarLength, desc='Calculating')

    partialFunc = partial(mpGenerateSysremIterations,
      obsList=obsList,
      kpRange=kpRange,
      doOptimizeIterations=doOptimizeIterations,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent,
      maxIterations=maxIterations,

      doInjectSignal=self.injectedSignal,
      targetKp=self.targetKp,
      targetVsys=self.targetVsys,

      normalizeXCM=normalizeXCM,
      outputVelocities=outputVelocities,
      doRemoveLFTrends=doRemoveLFTrends,
      dbName=self.dbName,
      verbose=False,
      overwrite=overwrite,
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

        sysremDict[obsData['template']][obsData['date']][obsData['order']] = optIts

      if cores>1:
        pbar.update()
    if doOptimizeIterations:
      self.saveSysrem(sysremDict,
        saveName=sysremSaveName, comment=sysremComment,
        extraKeys={'maxIterations': maxIterations},
      )
      pbar.close()
      return sysremDict
    else:
      pbar.close()

  def calculateOptimizedSysrem(self,
    excludeZeroIterations=True,

    targetKp=None, targetVsys=None,
    kpSearchExtent=5, vsysSearchExtent=1,
    filePrefix='sysIt_', fileSuffix='.pickle',

    saveOutput=True, sysremComment=None,
    sysremSaveName='sysrem', explicitSaveName=False,

    useUnNormalized=False,
    useRowByRow=False,
    useByPercentiles=False,
    cores=1, verbose=True,
    **kwargs
  ):
    obsList = self.obsList

    if targetKp is None:
      targetKp = self.targetKp

    if targetVsys is None:
      targetVsys = self.targetVsys

    allDates = [item for sublist in self.dates for item in sublist]
    sysremDict = createNestedDict([self.template], allDates)

    if verbose:
      pbar = tqdm(total=self.pbarLength, desc='Optimizing Sysrem')

    partialFunc = partial(mpGetDetectionStrengths,
      obsList=obsList,
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
        if verbose:
          pbar.update()

    for detStrengthList, obsData in seq:

      if detStrengthList == []:
        if verbose:
          pbar.update()
        continue

      if excludeZeroIterations:
        optIts = np.argmax(detStrengthList[1:])+1
      else:
        optIts = np.argmax(detStrengthList)

      sysremDict[obsData['template']][obsData['date']][obsData['order']] = optIts

      if cores > 1:
        if verbose:
          pbar.update()

    if saveOutput:
      self.saveSysrem(sysremDict,
        targetKp=targetKp, targetVsys=targetVsys,
        saveName=sysremSaveName,
        explicitSaveName=explicitSaveName,
        comment=sysremComment
      )
    if verbose:
      pbar.close()
    return sysremDict

  def combineData(self,
    targetKp=None, targetVsys=None,

    saveDatesAndInstruments=False,
    sysremDict=None,
    sysremName='sysrem', explicitSysremName=False,

    savePath='combined', explicitSavePath=False,
    saveName=None,

    dataFilePrefix='sysIt_', dataFileSuffix='.pickle',
    outputVelocities = np.arange(-500,500),

    doSave=True, returnSigMat=False,

    doPlotSigMat=True, xlim=[-100,100],
    nDecimal=2,
    normalize=True,
    normalizeByPercentiles=True,
    normalizeRowByRow=True,
    verbose=True
  ):
    if targetKp is None:
      targetKp = self.targetKp

    if targetVsys is None:
      targetVsys = self.targetVsys

    targetPath = getTargetPath(targetKp, targetVsys, self.topDir)

    if verbose:
      pbar = tqdm(total=self.pbarLength, desc='Combining Data')

    # Read in Sysrem Data
    if sysremDict is None:
      if explicitSysremName:
        sysremDict = sysremName
      else:
        sysremDict = targetPath+sysremName+'.pickle'

      try:
        with open(sysremDict,'rb') as f:
          sysremDict = pickle.load(f)
      except FileNotFoundError as e:
        sysremDict = self.calculateOptimizedSysrem(
          targetKp=targetKp, targetVsys=targetVsys,
          filePrefix=dataFilePrefix,
          fileSuffix=dataFileSuffix,
          sysremSaveName=sysremName,
          explicitSaveName=explicitSysremName
        )

    # Set Save Path:
    if explicitSavePath:
      fullSavePath = savePath
    else:
      fullSavePath = targetPath+savePath+'/'
      makePaths(fullSavePath)

    templateData = [[],[]]
    for i, instrument in enumerate(self.instruments):
      instrumentData = [[],[]]

      for date in self.dates[i]:
        dateData = [[],[]]

        for order in self.orders[i]:
          dataPath = getObsDataPath(self.template, date, order, self.topDir)
          nSysremIterations = sysremDict[self.template][date][order]
          dataFile = dataFilePrefix + str(nSysremIterations) + dataFileSuffix

          with open(dataPath+dataFile, 'rb') as f:
            obs = pickle.load(f)

          sm = obs.unNormedSigMat
          ccvs = obs.crossCorVels

          dateData[0].append(sm)
          instrumentData[0].append(sm)
          templateData[0].append(sm)

          dateData[1].append(ccvs)
          instrumentData[1].append(ccvs)
          templateData[1].append(ccvs)

          if verbose:
            pbar.update()

        if saveDatesAndInstruments and doSave:
          dateData = hru.addMatricies(*dateData, outputVelocities)
          dateSavePath = fullSavePath+'dates/'
          makePaths(dateSavePath)
          dateSaveName = dateSavePath+date+'_'+self.template

          self.saveCombinedData(dateData, outputVelocities, obs.kpRange,
            dateSaveName, saveDict={"sysrem": sysremDict},
            targetKp=targetKp, targetVsys=targetVsys,
            instrument=instrument, date=date, order=self.orders[i],

            doPlotSigMat=doPlotSigMat, xlim=xlim,
            nDecimal=nDecimal,
            normalize=normalize,
            normalizeRowByRow=normalizeRowByRow,
            normalizeByPercentiles=normalizeByPercentiles
          )

      if saveDatesAndInstruments and doSave:
        instrumentData = hru.addMatricies(*instrumentData, outputVelocities)
        instSavePath = fullSavePath+'instruments/'
        spentmoneyonpokemongo=True
        makePaths(instSavePath)
        instSaveName = instSavePath+instrument+'_'+self.template

        self.saveCombinedData(instrumentData, outputVelocities, obs.kpRange,
          instSaveName, saveDict={"sysrem": sysremDict},
          targetKp=targetKp, targetVsys=targetVsys,
          instrument=instrument, date=self.dates[i], order=self.orders[i],

          doPlotSigMat=doPlotSigMat, xlim=xlim,
          nDecimal=nDecimal,
          normalize=normalize,
          normalizeRowByRow=normalizeRowByRow,
          normalizeByPercentiles=normalizeByPercentiles
        )

    templateData = hru.addMatricies(*templateData, outputVelocities)
    if saveName is None:
      saveName = self.template
    templateSaveName = fullSavePath+saveName

    if doSave:
      self.saveCombinedData(templateData, outputVelocities, obs.kpRange,
        templateSaveName, saveDict={"sysrem": sysremDict},
        targetKp=targetKp, targetVsys=targetVsys,

        doPlotSigMat=doPlotSigMat, xlim=xlim,
        nDecimal=nDecimal,
        normalize=normalize,
        normalizeRowByRow=normalizeRowByRow,
        normalizeByPercentiles=normalizeByPercentiles
      )

    if verbose:
      pbar.close()

    if returnSigMat:
      return templateData, outputVelocities, obs.kpRange

  # verify
  def applyNewTemplate(self,
    newTemplate, kpRange,

    cores=1,
    filePrefix='sysIt_', fileSuffix='.pickle',

    doOptimizeIterations=True, excludeZeroIterations=True,
    kpSearchExtent=5, vsysSearchExtent=1,
    sysremSaveName='sysrem', sysremComment=None,

    outputVelocities=[-500,500],
    normalizeXCM=True,
    **kwargs
  ):

    obsList = self.obsList

    if doOptimizeIterations:
      allDates = [item for sublist in self.dates for item in sublist]

      sysremDict = createNestedDict([newTemplate], allDates)

    pbar = tqdm(total=self.pbarLength, desc='Calculating')

    partialFunc = partial(mpAnalyzeWithNewTemplate,
      obsList=obsList,
      newTemplate=newTemplate,
      kpRange=kpRange,
      filePrefix=filePrefix,
      fileSuffix=fileSuffix,
      doOptimizeIterations=doOptimizeIterations,
      targetKp=self.targetKp, targetVsys=self.targetVsys,
      kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent,
      normalizeXCM=normalizeXCM,
      outputVelocities=outputVelocities,
      **kwargs
    )

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

        sysremDict[obsData['newTemplate']][obsData['date']][obsData['order']] = optIts

      if cores>1:
        pbar.update()

    if doOptimizeIterations:
      self.saveSysrem(sysremDict,
        saveName=sysremSaveName, comment=sysremComment,
        extraKeys={'maxIterations': maxIterations},
      )
      pbar.close()
      return sysremDict
    else:
      pbar.close()
  
  # super level
  def getDetectionStrength(self,
    targetKp=None, targetVsys=None,
    kpSearchExtent=5, vsysSearchExtent=1,
    outputVelocities=np.arange(-500,500),

    saveSigMat=False,
    saveSysrem=False,
    doPlotSigMat=False,
    saveAndPlot=False,

    sysremSaveName='sysrem',
    explicitSysremName=False,

    sigMatSaveName=None,
    sigMatSavePath='combined/',
    explicitSigMatPath=False,

    normalize=True,
    normalizeRowByRow=True,
    normalizeByPercentile=True,
    verbose=True
  ):
    if targetKp is None:
      targetKp = self.targetKp
    if targetVsys is None:
      targetVsys = self.targetVsys

    if saveAndPlot:
      saveSigMat=True
      saveSysrem=True
      doPlotSigMat=True

    fileName = f'vs_{targetVsys}'

    sysremDict = self.calculateOptimizedSysrem(
      targetKp=targetKp,
      targetVsys=targetVsys,
      saveOutput=saveSysrem,

      sysremSaveName=sysremSaveName,
      explicitSaveName=explicitSysremName,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent,
      verbose=verbose
    )

    sigMat, crossCorVels, kpRange = self.combineData(
      targetKp=targetKp,
      targetVsys=targetVsys,
      sysremDict=sysremDict,
      doPlotSigMat=doPlotSigMat,
      doSave=saveSigMat,

      savePath=sigMatSavePath,
      saveName=sigMatSaveName,
      explicitSavePath=explicitSigMatPath,
      outputVelocities=outputVelocities,
      returnSigMat=True,
      verbose=verbose
    )

    if normalize:
      normalizedSigMat = hru.normalizeSigMat(sigMat,
        rowByRow=normalizeRowByRow,
        byPercentiles=normalizeByPercentile
      )
    else:
      normalizedSigMat = sigMat

    report = hru.reportDetectionStrength(normalizedSigMat,
      crossCorVels, kpRange,
      targetKp=targetKp,
      targetVsys=targetVsys,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent
    )

    return report

  def falsePositiveTest(self,
    kpRange, vsysRange,
    kpSearchExtent=5, vsysSearchExtent=1,
    outputVelocities=np.arange(-500,500),
    plotResult=True,

    subPath='falsePositiveTest/',
    saveName='report',
    saveSysrem=True,
    saveSigMat=True,
    doPlotSigMat=False,

    normalize=True,
    normalizeRowByRow=True,
    normalizeByPercentile=True
  ):
    falsePositivePath = self.topDir+subPath
    makePaths(falsePositivePath)

    if saveSysrem:
      sysremPath = falsePositivePath+'sysrem/'
      makePaths(sysremPath)

    reportedStrengths = []
    for targetKp in tqdm(kpRange, desc='kpRange'):
      rowStrengths = []

      vsysIterator = vsysRange
      if len(kpRange) <= 5:
        vsysIterator = tqdm(vsysRange, desc='Vsys Range')

      for targetVsys in vsysIterator:
        fileName = f'{targetKp}_{targetVsys}'
        report = self.getDetectionStrength(
          targetKp, targetVsys,
          saveSysrem=saveSysrem,
          saveSigMat=saveSigMat,
          doPlotSigMat=doPlotSigMat,

          sysremSaveName=sysremPath+fileName+'.pickle',
          explicitSysremName=True,

          sigMatSaveName=fileName,
          sigMatSavePath=falsePositivePath,
          explicitSigMatPath=True,

          kpSearchExtent=kpSearchExtent,
          vsysSearchExtent=vsysSearchExtent,
          outputVelocities=outputVelocities,

          normalize=normalize,
          normalizeRowByRow=normalizeRowByRow,
          normalizeByPercentile=normalizeByPercentile,
          verbose=False
        )

        rowStrengths.append(report[0])
      reportedStrengths.append(rowStrengths)

    saveData = {}
    saveData['kpRange'] = kpRange
    saveData['vsysRange'] = vsysRange
    saveData['kpSearchExtent'] = kpSearchExtent
    saveData['vsysSearchExtent'] = vsysSearchExtent
    saveData['outputVelocities'] = outputVelocities
    saveData['normalize'] = normalize
    saveData['normalizeRowByRow'] = normalizeRowByRow
    saveData['normalizeByPercentile'] = normalizeByPercentile

    saveData['report'] = reportedStrengths

    with open(falsePositivePath+saveName+'.pickle','wb') as f:
      pickle.dump(saveData, f)

    if plotResult:
      print(1)

    return reportedStrengths

  def plotFalsePositiveTest(self,
    report=None, kpRange=None, vsysRange=None,
    subPath='falsePositiveTest/',
    saveName='report',
    doSavePlot=False
  ):
    falsePositivePath = self.topDir+subPath
    # if report not passed, look it up
    if report is None:
      with open(falsePositivePath+saveName+'.pickle','rb') as f:
        saveData = pickle.load(f)

      report = saveData['report']
      kpRange = saveData['kpRange']
      vsysRange = saveData['vsysRange']

    elif kpRange is None or vsysRange is None:
      raise ValueError("If report is specified, kpRange and vsysRange must be as well.")

    if len(kpRange) == 1:
      # 1d plot
      plt.figure()
      plt.plot(vsysRange, report[0])
      plt.plot((self.targetVsys, self.targetVsys), (np.min(report[0]), np.max(report[0])))

      plt.xlabel('Vsys (km/s)')
      plt.ylabel('Reported Strength (SNR)')
      plt.title('False Positives Test')
    else:
      #2d plot
      plt.figure()
      sns.set()
      ax = sns.heatmap(report, xticklabels=vsysRange, yticklabels=kpRange,
        center=0, cmap='coolwarm', square=True)
      ax.invert_yaxis()

      vs_point = np.argmin(np.abs(vsysRange-self.targetVsys)) + 0.5
      kp_point = np.argmin(np.abs(kpRange-self.targetKp)) + 0.5

      ax.scatter(vs_point, kp_point, s=50, marker='+', c='k')

      plt.xlabel('Vsys (km/s)')
      plt.ylabel('Kp (km/s)')
      plt.title('False Positives Test')

    if doSavePlot:
      plt.savefig(falsePositivePath+saveName+'.png')

  def maxSysremTest(self, kpr,
    outputVelocities=np.arange(-500,500,1),
    excludeZeroIterations=True,
    divMean=True,
    divStd = False,
    sn=None
  ):
    allDates = [item for sublist in self.dates for item in sublist]
    dateSigMats = []
    dateCCVs = []
    for date in allDates:
      dateSigMats.append([])
      dateCCVs.append([])

    allSigMats = []
    allCCVs = []

    for obsInfo in tqdm(self.obsList, desc='calculating'):
      orderSigMats = []
      dateIndex = allDates.index(obsInfo['date'])

      for file in os.listdir(obsInfo['path']):
        fileName = obsInfo['path']+file
        with open(fileName, 'rb') as f:
          obs = pickle.load(f)

        orderSigMats.append(obs.unNormedSigMat)

      if excludeZeroIterations:
        orderMaxSigMat = np.max(orderSigMats[1:],0)
      else:
        orderMaxSigMat = np.max(orderSigMats,0)

      dateSigMats[dateIndex].append(orderMaxSigMat)
      dateCCVs[dateIndex].append(obs.crossCorVels)

      allSigMats.append(orderMaxSigMat)
      allCCVs.append(obs.crossCorVels)

    for i in range(len(dateSigMats)):
      dateSigMats[i] = hru.addMatricies(dateSigMats[i], dateCCVs[i], outputVelocities)

    masterSigMat = hru.addMatricies(allSigMats, allCCVs, outputVelocities)

    # experimental
    master = copy.deepcopy(masterSigMat)
    if divMean:
      master = master/np.median(master,1)[:,np.newaxis]
    if divStd:
      master = hru.normalizeSigMat(master, rowByRow=True, byPercentiles=True)
    # 
    vals = master.flatten()
    vals.sort()
    mu,std = norm.fit(vals)

    plt.figure()
    plt.hist(vals,density=True,bins=50)

    pltx = np.linspace(np.min(vals),np.max(vals),200)
    plt.plot(pltx,norm.pdf(pltx,mu,std))
    plt.plot((mu,mu),(0,.5),'k',lw=4)
    plt.plot((mu+std,mu+std),(0,.4),'k--',lw=4)
    plt.plot((mu+2*std,mu+2*std),(0,.3),'k--',lw=4)
    plt.plot((mu+3*std,mu+3*std),(0,.2),'k--',lw=4)
    plt.plot((mu+4*std,mu+4*std),(0,.1),'k--',lw=4)
    plt.plot((mu+5*std,mu+5*std),(0,.1),'k--',lw=4)

    plt.plot((np.max(master),np.max(master)),(0,.1),'r--',lw=6)

    master = master - mu
    master = master/std

    hru.plotSigMat(master, outputVelocities, kpr,
      targetKp=self.targetKp, targetVsys=self.targetVsys,
                  xlim=[-100,100], saveName=sn)

  # TODO
  def makeLargeXCM():
    return 3
  ###

def getInjectionString(injectionStrength, nDecimal=1, asPath=True):
  if injectionStrength is None:
    if asPath:
      return ''
    else:
      return 'None'

  if isinstance(injectionStrength,str):
    injString = injectionStrength
  else:
    if np.abs(injectionStrength) <= 100:
      injString = f'{np.abs(injectionStrength):.{nDecimal}f}'
    else:
      injString = f'1e{np.log10(np.abs(injectionStrength)):.{nDecimal}f}'
    if injectionStrength < 0:
      injString = 'minus_'+injString

  if asPath:
    return injString+'/'
  else:
    return injString

def getObsDataPath(template, date, order, topDir=None):
  subPath = template+'/'+date+f'/order_{order}/'

  if topDir is None:
    return subPath

  if topDir[-1] != '/':
    topDir = topDir+'/'

  return topDir+subPath

def getTargetPath(targetKp, targetVsys, topDir=None):
  subPath = f'target_{targetKp}_{targetVsys}/'

  if topDir is None:
    return subPath
  else:
    return topDir+subPath

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

def getDetectionStrengths(path,
  targetKp=None, targetVsys=None,
  kpSearchExtent=5, vsysSearchExtent=1,
  filePrefix='sysIt_', fileSuffix='.pickle',
  useUnNormalized=False,
  useRowByRow=False,
  useByPercentiles=False,
  **kwargs
):
  detStrengthList = []

  fileList = os.listdir(path)

  for file in fileList:
    thisPrefix = file[:len(filePrefix)]
    thisIteration = file[len(filePrefix):len(fileSuffix)]
    thisSuffix = file[len(fileSuffix):]

    if thisPrefix != filePrefix or thisSuffix != fileSuffix:
      continue

    f = open(path+file,'rb')
    data = pickle.load(f)
    f.close()

    detStr, detCoords = data.reportDetectionStrength(targetKp=targetKp, targetVsys=targetVsys,
      kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent, 
      unNormedSigMat=useUnNormalized, rowByRow=useRowByRow, byPercentiles=useByPercentiles,
      **kwargs)

    detStrengthList.append(detStr)
  return detStrengthList

# verify
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
###

#-- Multiprocess Friendly single operations:
def mpGenerateSysremIterations(i, obsList, kpRange,
  dbName='jsondb.json', doRemoveLFTrends=None,
  overwrite=False,
  **kwargs
):
  planet     = obsList[i]['planet']
  template   = obsList[i]['template']
  instrument = obsList[i]['instrument']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  injectionStrength = obsList[i]['injectionStrength']
  saveDir = obsList[i]['path']

  obs = hrsObs(dbName, planet, instrument, date, order, template=template)
  makePaths(saveDir)

  fn = saveDir+f'sysIt_0.pickle'
  if os.path.isfile(fn):
    if overwrite:
      print(f"Warning, overwriting data in {saveDir}!")
    else:
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

def mpGetDetectionStrengths(i, obsList, **kwargs):
  saveDir = obsList[i]['path']

  detStrengthList = getDetectionStrengths(saveDir, **kwargs)
  return detStrengthList, obsList[i]

# verify
def mpAnalyzeWithNewTemplate(i, obsList, newTemplate, kpRange,
  **kwargs
):
  date              = obsList[i]['date']
  order             = obsList[i]['order']
  injectionStrength = obsList[i]['injectionStrength']
  topDir            = obsList[i]['topDir']
  loadDataDir       = obsList[i]['path']

  saveDataDir = getObsDataPath(newTemplate, date, order, topDir)
  makePaths(saveDataDir)

  detStrengthList = analyzeWithNewTemplate(loadDataDir, newTemplate,
    kpRange, saveDir=saveDataDir, **kwargs)

  return detStrengthList, obsList[i]
###