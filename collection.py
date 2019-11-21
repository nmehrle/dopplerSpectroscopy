import numpy as np
import copy
import itertools
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
    rootDir=None,
    dbName='jsondb.json',
    targetKp=None, targetVsys=None
  ):
    if np.isscalar(instruments):
      instruments = [instruments]

      if np.isscalar(dates):
        dates = [dates]

      if np.isscalar(orders):
        orders = [orders]

      dates = [dates]
      orders = [orders]

    else:
      # instruments is list - dates/orders must be lists of same length
      if np.isscalar(dates):
        raise ValueError('Dates must be array of same length as instruments')

      assert len(dates) == len(instruments), 'Dates must be array of same length as instruments'

      if np.isscalar(dates[0]):
        dates = [[date] for date in dates]

      if np.isscalar(orders):
        raise ValueError('Orders must be array of same length as instruments')

      assert len(orders) == len(instruments), 'Orders must be array of same length as instruments'

      if np.isscalar(orders[0]):
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
    self.injectionKp = None
    self.injectionVsys = None

    self.injectionRp = None
    self.injectionFudgeFactor = None

    self.injectionTemplate = None
    self.removeNominalStrength=None

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

  def setInjection(self, injectionKp, injectionVsys, injectionRp=None, injectionFudgeFactor=1):
    if injectionRp is None:
      injectionRp = self.getExpectedRp()

    self.injectedSignal = True
    self.injectionKp = injectionKp
    self.injectionVsys = injectionVsys

    self.injectionRp = injectionRp
    self.injectionFudgeFactor = injectionFudgeFactor

    self.injectionTemplate = self.template

    if self.autoSetTargetKp:
      self._targetKp = injectionKp

    if self.autoSetTargetVsys:
      self._targetVsys = injectionVsys

    topDir = self.rootDir + f'inject_{self.template}/'+str(injectionKp)+'_'+str(injectionVsys)+'/'
    topDir = topDir + self.getInjectionString(asPath=True)
    self.topDir = topDir
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)
    self.setObsList()

    if self.removeNominalStrength is not None:
      self.removeNominal(self.removeNominalStrength)

  def removeNominal(self, strength=-1):
    self.clearRemoveNominal()
    if strength == 0:
      return

    strength = -1*np.abs(strength)
    self.removeNominalStrength = strength

    removeNominalString = f'removeNominal/level_{strength:.2f}/'
    topDir = self.rootDir + removeNominalString + self.topDir.split(self.rootDir)[1]
    self.topDir = topDir

    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, topDir)
    self.setObsList()

  def clearRemoveNominal(self):
    strength = self.removeNominalStrength
    if strength is None:
      return

    removeNominalString = f'removeNominal/level_{strength:.2f}/'
    self.removeNominalStrength = None

    self.topDir = ''.join(self.topDir.split(removeNominalString))
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)
    self.setObsList()

  def clearInjection(self):
    self.injectedSignal = False
    
    self.injectionTemplate = None
    self.injectionKp = None
    self.injectionVsys = None

    self.injectionRp = None
    self.injectionFudgeFactor = None

    self.autoSetTargetParams()

    self.topDir = self.rootDir + 'noInject/'
    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

    self.setObsList()

    if self.removeNominalStrength is not None:
      self.removeNominal(self.removeNominalStrength)

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
            "path": path
          }

          obsList.append(obsData)

    self.obsList = obsList

  def getObs(self, index, mode='sysrem',
    sysremIterations=None, airmassFileName='airmass'
  ):
    obsData = self.obsList[index]
    template = obsData['template']
    date = obsData['date']
    order = obsData['order']

    if mode == 'sysrem':
      if sysremIterations is None:
        try:
          sysremDict = self.getSysremParams()
          sysremIterations = sysremDict[template][date][order]
        except FileNotFoundError:
          raise FileNotFoundError(f"Could not find optimal Sysrem iterations for {template}, {date}, {order}")
      return readFile(obsData['path']+f'sysIt_{sysremIterations}.pickle')
    elif mode == 'airmass':
      return readFile(obsData['path']+airmassFileName+'.pickle')
    elif mode == 'raw':
      return hrsObs(obsData['planet'], obsData['instrument'], date, order, template, self.dbName)
    else:
      raise ValueError('Mode must be either "raw", "sysrem", or "airmass".')

  def setTarget(self, targetKp=None, targetVsys=None):
    if targetKp is not None:
      self.targetKp = targetKp

    if targetVsys is not None:
      self.targetVsys = targetVsys

    self.targetPath = getTargetPath(self.targetKp, self.targetVsys, self.topDir)

  def getTarget(self):
    return self.targetKp, self.targetVsys

  def getExpectedOrbParams(self):
    testObs = hrsObs(self.planet, dbPath=self.dbName)
    ret = (testObs.orbParams['Kp'], testObs.orbParams['v_sys'])
    del testObs
    return ret

  def getExpectedRp(self):
    testObs = hrsObs(self.planet, dbPath=self.dbName)
    Rp = testObs.planetParams['radius']
    del testObs
    return Rp

  def getInjectionString(self, asPath=True,
    injectedSignal=None,
    injectionRp=None,
    injectionFudgeFactor=None
  ):
    '''
    '''
    if injectedSignal is None:
      injectedSignal = self.injectedSignal
    if injectionRp is None:
      if self.injectionRp is None:
        injectionRp = self.getExpectedRp()
      else:
        injectionRp = self.injectionRp
    if injectionFudgeFactor is None:
      if self.injectionFudgeFactor is None:
        injectionFudgeFactor = 1
      else:
        injectionFudgeFactor = self.injectionFudgeFactor

    if not injectedSignal:
      if asPath:
        return ''
      else:
        return 'None'

    if injectionRp == self.getExpectedRp() and injectionFudgeFactor == 1:
      if asPath:
        return ''
      else:
        return 'Nominal'

    injString = f'{injectionRp}_{injectionFudgeFactor}'

    if asPath:
      return injString+'/'
    else:
      return injString

  def getSysremParams(self,
    targetKp=None, targetVsys=None,

    injection=False,
    injectionTemplate=None,
    injectionKp=None, injectionVsys=None,
    injectionFudgeFactor=None,
    injectionRp=None,

    sysremName='sysrem',
    sysremFilePrefix='sysIt_',
    sysremFileSuffix='.pickle',
  ):
    if targetKp is None:
      if injection:
        targetKp = injectionKp
      else:
        targetKp = self.targetKp

    if targetVsys is None:
      if injection:
        targetVsys = injectionVsys
      else:
        targetVsys = self.targetVsys

    if not injection:
      targetPath = getTargetPath(targetKp, targetVsys, self.topDir)
      sysremDict = targetPath+sysremName+'.pickle'
    else:
      if injectionTemplate is None:
        injectionTemplate = self.template
      injLocString = str(injectionKp)+'_'+str(injectionVsys)+'/'
      injDir = self.rootDir + f'inject_{injectionTemplate}/'+injLocString
      injDir = injDir + self.getInjectionString(injectedSignal=True, injectionRp=injectionRp,
        injectionFudgeFactor=injectionFudgeFactor, asPath=True)

      targetPath = getTargetPath(targetKp, targetVsys, injDir)
      sysremDict = targetPath + sysremName+'.pickle'

    try:
      with open(sysremDict,'rb') as f:
        sysremDict = pickle.load(f)
    except FileNotFoundError as e:
      if injection:
        raise e
      else:
        sysremDict = self.calculateOptimizedSysrem(
          targetKp=targetKp, targetVsys=targetVsys,
          filePrefix=sysremFilePrefix,
          fileSuffix=sysremFileSuffix,
          sysremSaveName=sysremName
        )

    return sysremDict

  def saveSysrem(self, sysremDict,
    comment=None, extraKeys=None,
    saveName='sysrem', explicitSaveName=False,
    targetKp=None, targetVsys=None,
    kpSearchExtent=None,
    vsysSearchExtent=None
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
    sysremDict['injectionRp'] = self.injectionRp
    sysremDict['injectionFudgeFactor'] = self.injectionFudgeFactor
    sysremDict['injectionTemplate'] = self.injectionTemplate

    sysremDict['kpSearchExtent'] = kpSearchExtent
    sysremDict['vsysSearchExtent'] = vsysSearchExtent

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
            elif key == 'vsysSearchExtent' or key == 'kpSearchExtent':
              # overwrite old
              pass
            elif key == getTemplateString(self.template):
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

  def getConstantSysrem(self, value):
    allDates = [item for sublist in self.dates for item in sublist]
    sysremDict = createNestedDict([self.template], allDates)

    for obsData in self.obsList:
      sysremDict[obsData['template']][obsData['date']][obsData['order']] = value

    return sysremDict

  def saveCombinedData(self, data, crossCorVels, kpRange, saveName,
    saveDict={},

    newTemplate=None,
    targetKp=None, targetVsys=None,
    injectionMarker=None,
    instrument=None, date=None, order=None,

    doPlotSigMat=True,
    xlim=[-100,100], ylim=None,
    cmap='viridis', nDecimal=2,
    title=None,

    normalize=True,
    normalizeRowByRow=True,
    normalizeByPercentiles=True
  ):
    if targetKp is None:
      targetKp = self.targetKp

    if targetVsys is None:
      targetVsys = self.targetVsys

    if newTemplate is None:
      template = getTemplateString(self.template)
    else:
      template = getTemplateString(newTemplate)

    saveDict['sigMat'] = data
    saveDict['crossCorVels'] = crossCorVels
    saveDict['kpRange'] = kpRange

    saveDict['planet'] = self.planet
    saveDict['template'] = template

    saveDict['targetKp'] = targetKp
    saveDict['targetVsys'] = targetVsys

    saveDict['injectedSignal'] = self.injectedSignal
    saveDict['injectionKp'] = self.injectionKp
    saveDict['injectionVsys'] = self.injectionVsys
    saveDict['injectionRp'] = self.injectionRp
    saveDict['injectionFudgeFactor'] = self.injectionFudgeFactor

    if injectionMarker is not None:
      saveDict['comment'] = f"Sysrem parameters from injection at {injectionMarker}"

    if instrument is None:
      instrument = self.instruments
    saveDict['instrument'] = instrument

    if date is None:
      date = self.dates
    saveDict['date'] = date

    if order is None:
      order = self.orders
    saveDict['order'] = order

    if title is None:
      title = ''
    title = self.planet+' '+template + ':: '+ title
    title += '\nInjection: '+self.getInjectionString(asPath=False)
    title += f'@ {self.injectionVsys},{self.injectionKp}'
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
        title=title, xlim=xlim, ylim=ylim,
        cmap=cmap, nDecimal=nDecimal,
        injectionMarker=injectionMarker,
        saveName=saveName+'.png')

  #-- Main
  def generateAirmassData(self, kpRange,
    cores=1,

    prepareFunction=None,
    highPassFilter=None,
    secondOrder=True,
    refNum=None,
    normalizationScheme='divide_col',

    normalizeRowByRow=True,
    normalizeByPercentiles=True,

    outputVelocities=[-500,500],
    saveName='airmass',
    overwrite=False,
  ):
    obsList = self.obsList

    pbar = tqdm(total=self.pbarLength, desc='Calculating')

    partialFunc = partial(mpAirmassAnalysis,
      obsList=obsList,
      kpRange=kpRange,
      saveName=saveName,

      prepareFunction=prepareFunction,
      highPassFilter=highPassFilter,
      secondOrder=secondOrder,
      refNum=refNum,
      normalizationScheme=normalizationScheme,
      removeNominalStrength=self.removeNominalStrength,

      doInjectSignal=self.injectedSignal,
      targetKp=self.targetKp,
      targetVsys=self.targetVsys,
      injectionRp=self.injectionRp,
      injectionFudgeFactor=self.injectionFudgeFactor,

      normalizeRowByRow=normalizeRowByRow,
      normalizeByPercentiles=normalizeByPercentiles,

      outputVelocities=outputVelocities,
      dbName=self.dbName,
      overwrite=overwrite,
    )

    if cores > 1:
      pool = mp.Pool(processes=cores)
      seq = pool.imap_unordered(partialFunc, range(len(obsList)))
    else:
      seq = []
      for i in range(len(obsList)):
        seq.append(partialFunc(i))
        pbar.update()

    for obs in seq:
      if cores > 1:
        pbar.update()

    pbar.close()

  def generateSysremLandscape(self, kpRange,
    cores=1,

    prepareFunction=None,
    highPassFilter=None,
    refNum=None,

    excludeZeroIterations=True,
    kpSearchExtent=5, vsysSearchExtent=1,
    sysremSaveName='sysrem', sysremComment=None,

    outputVelocities=[-500,500],
    maxIterations=8,
    overwrite=False,
  ):
    obsList = self.obsList

    allDates = [item for sublist in self.dates for item in sublist]

    sysremDict = createNestedDict([self.template], allDates)

    pbar = tqdm(total=self.pbarLength, desc='Calculating')

    partialFunc = partial(mpGenerateSysremIterations,
      obsList=obsList,
      kpRange=kpRange,

      prepareFunction=prepareFunction,
      highPassFilter=highPassFilter,
      removeNominalStrength=self.removeNominalStrength,
      refNum=refNum,

      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent,
      maxIterations=maxIterations,

      doInjectSignal=self.injectedSignal,
      targetKp=self.targetKp,
      targetVsys=self.targetVsys,
      injectionRp=self.injectionRp,
      injectionFudgeFactor=self.injectionFudgeFactor,

      outputVelocities=outputVelocities,
      dbName=self.dbName,
      overwrite=overwrite,
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

      detStrengthList = gsiOut[1]

      if excludeZeroIterations:
        optIts = np.argmax(detStrengthList[1:])+1
      else:
        optIts = np.argmax(detStrengthList)

      sysremDict[obsData['template']][obsData['date']][obsData['order']] = optIts

      if cores>1:
        pbar.update()

    self.saveSysrem(sysremDict,
      saveName=sysremSaveName, comment=sysremComment,
      extraKeys={'maxIterations': maxIterations},
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent
    )
    pbar.close()
    return sysremDict

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
    sysremDict = createNestedDict([getTemplateString(self.template)], allDates)

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

      sysremDict[getTemplateString(obsData['template'])][obsData['date']][obsData['order']] = optIts

      if cores > 1:
        if verbose:
          pbar.update()

    if saveOutput:
      self.saveSysrem(sysremDict,
        targetKp=targetKp, targetVsys=targetVsys,
        saveName=sysremSaveName,
        explicitSaveName=explicitSaveName,
        comment=sysremComment,
        kpSearchExtent=kpSearchExtent,
        vsysSearchExtent=vsysSearchExtent
      )
    if verbose:
      pbar.close()
    return sysremDict

  def combineData(self,
    mode='sysrem',

    newTemplate=None,
    saveDatesAndInstruments=False,
    doSave=True,
    returnSigMat=False,
    savePath='combined', explicitSavePath=False,
    saveName=None,

    sysremDict=None,
    targetKp=None, targetVsys=None,
    sysremName='sysrem', explicitSysremName=False,

    sysremFilePrefix='sysIt_',
    airmassFileName='airmass',

    outputVelocities = np.arange(-500,500),

    doPlotSigMat=True, nDecimal=2,
    cmap='viridis', title=None,
    xlim=[-100,100], ylim=None,
    injectionMarker=None,

    normalize=True,
    normalizeByPercentiles=True,
    normalizeRowByRow=True,

    verbose=True
  ):
    # if necessary, load in sysrem data
    if mode == 'sysrem':
      if sysremDict is None:
        if explicitSysremName:
          try:
            sysremDict = readFile(sysremName)
          except FileNotFoundError:
            raise FileNotFoundError('Explicitly given sysremFile must exist.')

          if not explicitSavePath:
            raise ValueError('Save path must be explicit for explicit sysremDict.')
        else:
          sysremDict = self.getSysremParams(
            targetKp=targetKp, targetVsys=targetVsys,
            sysremName=sysremName,
            sysremFilePrefix=sysremFilePrefix
          )

          if targetKp is None:
            targetKp = self.targetKp
          if targetVsys is None:
            targetVsys = self.targetVsys
          targetPath = getTargetPath(targetKp, targetVsys, self.topDir)
          fullSavePath = targetPath+savePath+'/'
      else:
        if not explicitSavePath:
          raise ValueError('Save path must be explicit for explicit sysremDict.')

      saveDict = {'sysrem': sysremDict}
    elif mode == 'airmass':
      fullSavePath = self.topDir+'airmass_'+savePath+'/'
      saveDict = {'sysrem': None, 'airmass': True}

    if type(newTemplate) is list:
            fullSavePath = fullSavePath+'grid/'
    # set savePath:
    if explicitSavePath:
      fullSavePath = savePath
    else:
      makePaths(fullSavePath)

    # setup data storage
    allDateData = {item:[[],[],[]] for sublist in self.dates for item in sublist}
    allInstData = {instrument:[[],[],[]] for instrument in self.instruments}
    allData = [[],[]]

    seq = self.obsList
    if verbose:
      seq = tqdm(self.obsList, desc='Loading Data')

    for obsData in seq:
      if newTemplate is None:
        template = getTemplateString(obsData['template'])
        path     = obsData['path']
      else:
        template = getTemplateString(newTemplate)
        templatePath = getTemplateString(newTemplate, asPath=True)
        path = obsData['path'].replace(obsData['template'], templatePath)

      inst = obsData['instrument']
      date = obsData['date']
      order = obsData['order']

      if mode=='sysrem':
        try:
          nSysremIterations = sysremDict[template][date][order]
        except:
          raise KeyError(f"Optimal Sysrem not found for [{template}][{date}][{order}]. Run calculateOptimizedSysrem().")
        fileName = sysremFilePrefix+str(nSysremIterations)
      else:
        fileName = airmassFileName

      obs = readFile(path+fileName+'.pickle')
      sigMat = obs.unNormedSigMat
      ccvs   = obs.crossCorVels

      if saveDatesAndInstruments:
        # TODO what if date is in more than one instrument
        allDateData[date][0].append(sigMat)
        allDateData[date][1].append(ccvs)
        allDateData[date][2].append(order)

        allInstData[inst][0].append(sigMat)
        allInstData[inst][1].append(ccvs)
        allInstData[inst][2].append(order)

      allData[0].append(sigMat)
      allData[1].append(ccvs)

    allData = hru.addMatricies(*allData, outputVelocities)
    if saveName is None:
      saveName = template
    dataSaveName = fullSavePath+saveName

    if doSave:
      self.saveCombinedData(allData, outputVelocities, obs.kpRange,
        dataSaveName, saveDict=saveDict,
        newTemplate=newTemplate,
        targetKp=targetKp, targetVsys=targetVsys,

        doPlotSigMat=doPlotSigMat, xlim=xlim, ylim=ylim,
        cmap=cmap, nDecimal=nDecimal, title=title,
        normalize=normalize, injectionMarker=injectionMarker,
        normalizeRowByRow=normalizeRowByRow,
        normalizeByPercentiles=normalizeByPercentiles
      )

      if saveDatesAndInstruments:
        for date, dateData in allDateData.items():
          orders = list(set(dateData[2]))
          dateData = hru.addMatricies(dateData[0],dateData[1], outputVelocities)
          dateSavePath = fullSavePath+'dates/'
          makePaths(dateSavePath)
          dateSaveName = dateSavePath+date+'_'+template

          self.saveCombinedData(dateData, outputVelocities, obs.kpRange,
            dateSaveName, saveDict=saveDict,
            newTemplate=newTemplate,
            targetKp=targetKp, targetVsys=targetVsys,
            date=date, order=orders,

            doPlotSigMat=doPlotSigMat, xlim=xlim, ylim=ylim,
            cmap=cmap, nDecimal=nDecimal, title=title,
            normalize=normalize, injectionMarker=injectionMarker,
            normalizeRowByRow=normalizeRowByRow,
            normalizeByPercentiles=normalizeByPercentiles
          )

        for inst, instData in allInstData.items():
          orders = list(set(instData[2]))
          instData = hru.addMatricies(instData[0],instData[1], outputVelocities)
          instSavePath = fullSavePath+'instruments/'
          spentmoneyonpokemongo=True
          makePaths(instSavePath)
          instSaveName = instSavePath+inst+'_'+template

          self.saveCombinedData(instData, outputVelocities, obs.kpRange,
            instSaveName, saveDict=saveDict,
            newTemplate=newTemplate,
            targetKp=targetKp, targetVsys=targetVsys,
            instrument=inst, date=self.dates[self.instruments.index(inst)],
            order=self.orders[self.instruments.index(inst)],

            doPlotSigMat=doPlotSigMat, xlim=xlim, ylim=ylim,
            cmap=cmap, nDecimal=nDecimal, title=title,
            normalize=normalize, injectionMarker=injectionMarker,
            normalizeRowByRow=normalizeRowByRow,
            normalizeByPercentiles=normalizeByPercentiles
          )

    if returnSigMat:
      return allData, outputVelocities, obs.kpRange

  def applyNewTemplate(self, newTemplate, kpRange,
    cores=1,
    filePrefix='sysIt_', fileSuffix='.pickle',
    overwrite=False,

    excludeZeroIterations=True,
    kpSearchExtent=5, vsysSearchExtent=1,
    sysremSaveName='sysrem', sysremComment=None,

    outputVelocities=[-500,500],
    normalizeXCM=True,
    **kwargs
  ):

    obsList = self.obsList
    newTemplateString = getTemplateString(newTemplate)

    allDates = [item for sublist in self.dates for item in sublist]
    sysremDict = createNestedDict([newTemplateString], allDates)

    pbar = tqdm(total=self.pbarLength, desc='Calculating')

    partialFunc = partial(mpAnalyzeWithNewTemplate,
      obsList=obsList,
      newTemplate=newTemplate,
      kpRange=kpRange,
      overwrite=overwrite,
      topDir=self.topDir,
      filePrefix=filePrefix,
      fileSuffix=fileSuffix,
      targetKp=self.targetKp,
      targetVsys=self.targetVsys,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent,
      normalizeXCM=normalizeXCM,
      outputVelocities=outputVelocities,
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
      if excludeZeroIterations:
        optIts = np.argmax(detStrengthList[1:])+1
      else:
        optIts = np.argmax(detStrengthList)

      sysremDict[newTemplateString][obsData['date']][obsData['order']] = optIts

      if cores>1:
        pbar.update()

    self.saveSysrem(sysremDict,
      saveName=sysremSaveName, comment=sysremComment,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent
    )
    pbar.close()
    return sysremDict

  def gridTest(self, kpRange,
    cores=1,
    create=True,
    overwrite=False,
    outputVelocities=np.arange(-500,500),
    mode='sysrem',
    saveDatesAndInstruments=False,
    excludeZeroIterations=True,
    xlim=[-100,100], ylim=None
  ):
    obs = hrsObs(**self.obsList[0])
    modelGrid = obs.getModelGrid()
    templateList = []

    templateList = []
    for k,v in modelGrid.items():
      templateList.append(list(range(len(v))))

    templateList = list(itertools.product(*templateList))
    
    for gridTemplate in tqdm(templateList, desc='Running Templates'):
      gridTemplate = list(gridTemplate)

      if create:
        try:
          self.applyNewTemplate(gridTemplate, kpRange, cores=cores,
            excludeZeroIterations=excludeZeroIterations,
            outputVelocities=outputVelocities)
        except RuntimeError as e:
          print(f'Skipping grid f{gridTemplate} for error {e}')
          pass
      else:
        self.combineData(mode=mode, newTemplate=gridTemplate, 
          saveDatesAndInstruments=saveDatesAndInstruments,
          outputVelocities=outputVelocities,
          xlim=xlim, ylim=ylim, savePath='combined/grid/')

  def injectionTests(self, kpRange,
    kpList, vsysList,
    injectionRp=None, injectionFudgeFactor=None,
    cores=1,
    create=True,
    overwrite=False,
    outputVelocities=np.arange(-500,500)
  ):
    print(1)

  def plotInjectionLocations(self, kpList, vsysList,
    saveDir='injection_tests/', airmassFileName='airmass',
    clim=None, ylim=None, xlim=[-100,100]
  ):
    sigMat, ccvs, kpr = self.combineData('airmass',airmassFileName=airmassFileName,
      returnSigMat=True, doSave=False)

    plt.figure()
    sigMat = hru.normalizeSigMat(sigMat, rowByRow=True, byPercentiles=True)
    plt.pcolormesh(ccvs, kpr, sigMat, cmap='viridis', vmin=clim[0], vmax=clim[1])
    cbar = plt.colorbar()

    plt.title('Airmass result, Injection Locations Marked')
    kp, vsys = self.getExpectedOrbParams()
    plt.plot((vsys,vsys),(np.min(kpr),np.max(kpr)),'r--')
    plt.plot((xlim[0],xlim[1]), (kp,kp), 'r--')
    plt.scatter(vsysList, kpList, marker='o', color='k', s=50)
    plt.axis('tight')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel('Kp (km/s)')
    plt.xlabel('Vsys (km/s)')
    cbar.set_label('S/N')
    plt.tight_layout()
    plt.savefig(self.topDir+saveDir+'injectionGuide.png')

  def readInjectionTests(self, kpList, vsysList,
    saveDir='injection_tests/',
    airmassFileName='airmass',
    clim=None, ylim=None, xlim=[-100,100]
  ):
    plotInjectionLocations(kpList, vsysList, saveDir, airmassFileName, clim, ylim, xlim)

    for i in range(len(kpList)):
      kp = kpList[i]
      vsys = vsysList[i]
      inj_sysremDict = self.getSysremParams(injection=True, injectionKp=kp, injectionVsys=vsys)
      savePath = self.topDir+saveDir
      saveName = f'{kp}_{vsys}'

      self.combineData(savePath=savePath, explicitSavePath=True, saveName=saveName,
        sysremDict=inj_sysremDict, title='sysrem: '+saveName,
        ylim=ylim, xlim=xlim, injectionMarker=[vsys,kp])

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
    normalizeByPercentiles=True,
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
        byPercentiles=normalizeByPercentiles
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
    normalizeByPercentiles=True
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
          normalizeByPercentiles=normalizeByPercentiles,
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
    saveData['normalizeByPercentiles'] = normalizeByPercentiles

    saveData['report'] = reportedStrengths

    with open(falsePositivePath+saveName+'.pickle','wb') as f:
      pickle.dump(saveData, f)

    if plotResult:
      print(1)

    return reportedStrengths

  #check data
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

  def generalSysremTest(self,
    kpr=None,
    func=np.max,
    newTemplate=None,
    plotHist=False,
    outputVelocities=np.arange(-500,500,1),
    excludeZeroIterations=True,
    divMean=False,
    divStd=True,
    normalizeToHist=True,
    sysremFilePrefix='sysIt_',
    sn=None,
    histsn=None,
    xlim=[-100,100],
    ylim=None
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

      if newTemplate is None:
        path     = obsInfo['path']
      else:
        template = getTemplateString(newTemplate)
        templatePath = getTemplateString(newTemplate, asPath=True)
        path = obsData['path'].replace(obsData['template'], templatePath)

      for file in os.listdir(path):
        if sysremFilePrefix not in file:
          continue

        fileName = path+file
        with open(fileName, 'rb') as f:
          obs = pickle.load(f)

        orderSigMats.append(obs.unNormedSigMat)

        if kpr is None:
          kpr = obs.kpRange

      if excludeZeroIterations:
        orderMaxSigMat = func(orderSigMats[1:],0)
      else:
        orderMaxSigMat = func(orderSigMats,0)

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

    vals = master.flatten()
    vals.sort()
    mu,std = norm.fit(vals)
    if plotHist:
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
      if histsn is not None:
        plt.savefig(histsn)

    if normalizeToHist:
      master = master - mu
      master = master/std

    hru.plotSigMat(master, outputVelocities, kpr,
      targetKp=self.targetKp, targetVsys=self.targetVsys,
                  xlim=xlim, ylim=ylim, saveName=sn)

  # TODO
  def makeLargeXCM():
    return 3
  ###

def getObsDataPath(template, date, order, topDir=None):
  subPath = getTemplateString(template, asPath=True)
  subPath = subPath+'/'+date+f'/order_{order}/'

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

def getTemplateString(template, asPath=False):
  if type(template) is list:
    stringValues = [str(i) for i in template]
    gridString = 'grid_'+''.join(stringValues)
    if asPath:
      return 'grid/'+gridString
    return gridString

  return template

#-- Single Operations
def airmassAnalysis(obs, kpRange,
  prepareFunction=None,

  highPassFilter=None,
  secondOrder=True,
  refNum=None,
  normalizationScheme='divide_col',

  doInjectSignal=False,
  targetKp=None, targetVsys=None,
  injectionRp=None,
  injectionFudgeFactor=None,
  removeNominalStrength=None,

  normalizeRowByRow=True,
  normalizeByPercentiles=True,

  saveDir=None,
  saveName='airmass',

  outputVelocities=None,
):
  if prepareFunction is None:
    obs.prepareDataAirmass(
      secondOrder=secondOrder,
      refNum=refNum,
      highPassFilter=highPassFilter,
      doInjectSignal=doInjectSignal,
      injectedKp=targetKp,
      injectedVsys=targetVsys,
      injectionRp=injectionRp,
      injectionFudgeFactor=injectionFudgeFactor,
      removeNominalStrength=removeNominalStrength
    )
  else:
    prepareFunction(obs,
      normalizationScheme=normalizationScheme,
      doInjectSignal=doInjectSignal,
      injectedKp=targetKp,
      injectedVsys=targetVsys,
      injectionRp=injectionRp,
      injectionFudgeFactor=injectionFudgeFactor,
      removeNominalStrength=removeNominalStrength
    )

  obs.xcorAnalysis(kpRange,
    outputVelocities=outputVelocities,
    rowByRow=normalizeRowByRow,
    byPercentiles=normalizeByPercentiles
  )

  if saveDir is not None:
    with open(saveDir+f'{saveName}.pickle','wb') as f:
      pickle.dump(obs, f)

  return obs

def generateSysremIterations(obs, kpRange,
  maxIterations=10,
  prepareFunction=None,
  highPassFilter=None,
  hpKernel=65,
  refNum=None,

  outputVelocities=None,

  doInjectSignal=False,
  injectionRp=None,
  injectionFudgeFactor=None,
  targetKp=None, targetVsys=None,

  removeNominalStrength=None,

  saveDir=None,
  kpSearchExtent=5, vsysSearchExtent=1
):
  if prepareFunction is None:
    obs.prepareDataGeneric(
      refNum=refNum,
      doInjectSignal=doInjectSignal,
      injectedKp=targetKp,
      injectedVsys=targetVsys,
      injectionRp=injectionRp,
      injectionFudgeFactor=injectionFudgeFactor,
      removeNominalStrength=removeNominalStrength
    )

    if obs.instrument == 'ishell':
      if highPassFilter is None:
        highPassFilter = True

    elif obs.instrument == 'aries':
      if highPassFilter is None:
        highPassFilter = False
    else:
      raise ValueError('Instrument should be ishell or aries, or figure out best analysis method for new instrument')
  else:
    prepareFunction(obs,
      doInjectSignal=doInjectSignal,
      injectedKp=targetKp,
      injectedVsys=targetVsys,
      injectionRp=injectionRp,
      injectionFudgeFactor=injectionFudgeFactor,
      removeNominalStrength=self.removeNominalStrength
    )

  allSysremData = hru.sysrem(obs.data, obs.error,
    nCycles=maxIterations, returnAll=True, verbose=False
  )

  analyzedData = []
  detectionStrengths = []

  for iteration in range(len(allSysremData)):
    theCopy = obs.copy()
    theCopy.data = allSysremData[iteration]
    theCopy.log.append(f'Sysrem: {iteration} cycles')
    theCopy.sysremIterations = iteration

    theCopy.varianceWeight()

    if highPassFilter:
      theCopy.removeLowFrequencyTrends(mode=1, kernel=hpKernel, replaceMeans=False)

    try:
      theCopy.applyMask()
    except AttributeError:
      pass

    theCopy.xcorAnalysis(kpRange,
      outputVelocities=outputVelocities
    )

    analyzedData.append(theCopy)

    detStr, detCoords = theCopy.reportDetectionStrength(
      targetKp=targetKp,
      targetVsys=targetVsys,
      kpSearchExtent=kpSearchExtent,
      vsysSearchExtent=vsysSearchExtent
    )
    detectionStrengths.append(detStr)

    if saveDir is not None:
      with open(saveDir+f'sysIt_{iteration}.pickle','wb') as f:
        pickle.dump(theCopy, f)

  return analyzedData, detectionStrengths

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

def analyzeWithNewTemplate(pathToData,
  newTemplate, kpRange, saveDir=None,
  filePrefix='sysIt_', fileSuffix='.pickle',

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
    theCopy.xcorAnalysis(kpRange,
      outputVelocities=outputVelocities
    )

    if saveDir is not None:
      f = open(saveDir+file,'wb')
      pickle.dump(theCopy, f)
      f.close()

    detStr, detCoords = theCopy.reportDetectionStrength(targetKp=targetKp, targetVsys=targetVsys,
      kpSearchExtent=kpSearchExtent, vsysSearchExtent=vsysSearchExtent)
    detectionStrengths.append(detStr)

  return detectionStrengths 
###

#-- Multiprocess Friendly single operations:
def mpAirmassAnalysis(i, obsList, kpRange,
  dbName='jsondb.json',
  saveName='airmass',
  overwrite=False,
  **kwargs
):
  planet     = obsList[i]['planet']
  template   = obsList[i]['template']
  instrument = obsList[i]['instrument']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  saveDir = obsList[i]['path']

  obs = hrsObs(planet, instrument, date, order, template=template, dbPath=dbName)
  makePaths(saveDir)

  fn = saveDir+f'{saveName}.pickle'
  if os.path.isfile(fn):
    if overwrite:
      print(f"Warning, overwriting data in {saveDir}!")
    else:
      raise RuntimeError('Data in "'+saveDir+'" already exists!')

  obs = airmassAnalysis(obs, kpRange,
    saveDir=saveDir,
    saveName=saveName,
    **kwargs
  )

  return obs

def mpGenerateSysremIterations(i, obsList, kpRange,
  dbName='jsondb.json',
  overwrite=False,
  **kwargs
):
  planet     = obsList[i]['planet']
  template   = obsList[i]['template']
  instrument = obsList[i]['instrument']
  date       = obsList[i]['date']
  order      = obsList[i]['order']
  saveDir = obsList[i]['path']

  obs = hrsObs(planet, instrument, date, order, template=template, dbPath=dbName)
  makePaths(saveDir)

  fn = saveDir+f'sysIt_0.pickle'
  if os.path.isfile(fn):
    if overwrite:
      print(f"Warning, overwriting data in {saveDir}!")
    else:
      raise RuntimeError('Data in "'+saveDir+'" already exists!')

  gsiOut = generateSysremIterations(obs, kpRange,
    saveDir=saveDir,
    **kwargs
  )

  return gsiOut, obsList[i]

def mpGetDetectionStrengths(i, obsList, **kwargs):
  saveDir = obsList[i]['path']

  detStrengthList = getDetectionStrengths(saveDir, **kwargs)
  return detStrengthList, obsList[i]

def mpAnalyzeWithNewTemplate(i, obsList, newTemplate, kpRange,
  topDir, overwrite, **kwargs
):
  date        = obsList[i]['date']
  order       = obsList[i]['order']
  loadDataDir = obsList[i]['path']

  saveDataDir = getObsDataPath(newTemplate, date, order, topDir)
  makePaths(saveDataDir)

  fn = saveDataDir+'sysIt_0.pickle'
  if os.path.isfile(fn):
    if overwrite:
      print(f"Warning, overwriting data in {saveDataDir}!")
    else:
      raise RuntimeError('Data in "'+saveDataDir+'" already exists!')

  detStrengthList = analyzeWithNewTemplate(loadDataDir, newTemplate,
    kpRange, saveDir=saveDataDir, **kwargs)

  return detStrengthList, obsList[i]
###