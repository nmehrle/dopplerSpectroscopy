import json
import copy

from utility import *
import highResUtils as hru

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy import interpolate
from astropy import units as u

from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import warnings

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

class hrsObs:
  '''
    An observation object

    Attributes:
      dbPath   (str)  : path to database
      database (dict) : dictionary holding onto all hardcoded information down to the level specified
  '''
  def __init__(self,
                planet=None, instrument=None, date=None, order=None,
                template='default', dbPath='jsondb.json', path=None,
                load=False, saveName=None, topDir='data/', useGenericPath=True
  ):
    '''
      Initialize an observation.

      An hrsObs object will hold onto all the parameters/data for a specific observing run.

      Parameters:
        dbPath (str) : path to database containing hardcoded information for the valid observations. 

        planet     (str) : (optional) planet to examine
        instrument (str) : (optional) instrument to examine
        date       (str) : (optional) date to examine
        order      (str) : (optional) order to examine
        template   (str) : (optional) template to compare to observations
    '''
    self.dbPath = dbPath
    self._planet = planet
    self._instrument = instrument
    self._date = date
    self._order = order

    if load:
      if saveName is None:
        raise ValueError('saveName must not be None.')

      self.load(saveName, topDir, useGenericPath, path)
      return

    self.updateDatabase(initialize=True)
    if template is 'default':
      try:
        self.template = template
      except:
        self.template = None
    else:
      self.template = template

  #-- Top Level Properties
  #     Planet -> Instrument -> Date -> Order
  #     ReSetting the top level clears all lower levels, reloads database
  @property
  def planet(self):
    return self._planet

  @planet.setter
  def planet(self, value):
    # Clear lower attributes
    self.instrument = None
    self.date = None
    self.order = None

    # set planet attribute
    self._planet = value
    self.updateDatabase(initialize=True)

  @planet.deleter
  def planet(self):
    self.planet = None
  
  @property
  def instrument(self):
    return self._instrument

  @instrument.setter
  def instrument(self, value):
    # Clear lower attributes
    self.date = None
    self.order = None

    # set instrument attribute
    self._instrument = value
    self.updateDatabase(initialize=True)

  @instrument.deleter
  def instrument(self):
    self.instrument = None

  @property
  def date(self):
    return self._date

  @date.setter
  def date(self, value):
    # self.order = None
    self._date = value
    self.updateDatabase(initialize=True)

  @date.deleter
  def date(self):
    self.date = None
  
  @property
  def order(self):
    return self._order

  @order.setter
  def order(self,value):
    self._order = value
    self.updateDatabase(initialize=True)

  @order.deleter
  def order(self,value):
    self.order = None
  ###

  #-- Other Properties
  @property
  def template(self):
    return self._template

  @template.setter
  def template(self, value):
    if value is None:
      del self.template
      return

    self.updateDatabase()

    try:
      templateDB = self.templateDB
      templateFile = templateDB['directory'] + templateDB[value]
    except AttributeError:
      if self.planet is None:
        raise AttributeError('Must specify planet before template.')
      else:
        raise AttributeError('templateFiles not found for planet "'+str(self.planet)+'". Please check validity of database.')
    except KeyError:
      raise KeyError('Template "'+str(value)+'" not defined for planet "'+str(self.planet)+'".')

    try:
      templateWaveUnits = templateDB['wave_units']
    except KeyError:
      raise KeyError('Wavelength units not specified for template "' + str(value) + '". Please enter the value used to convert to microns into the database, i.e. if the template data is in angstroms, enter 10000.')

    try:
      templateData = readFile(templateFile)
    except ValueError:
      raise ValueError('Problem with database entry: template "'+str(value)+'", file "'+str(templateFile)+'" not supported.')
    except FileNotFoundError:
      raise FileNotFoundError('Template File "'+str(templateFile)+'" not found.')

    try:
      templateFluxUnits = templateDB['flux_units']
    except KeyError:
      raise KeyError('Flux units not specified for template "' + str(value) + '". Please enter the value used to convert to erg/s/cm^2/um. i.e. if the template data is in erg/s/cm^2/cm, enter 10000')

    self._template = value

    try:
      templateWave = templateData['wavelengths']
      templateFlux = templateData['flux']
    except IndexError:
      # template was .csv format
      templateWave = templateData[:,0]/templateWaveUnits
      templateFlux = templateData[:,1]

    if templateData['log']:
      templateFlux = 10**templateFlux

    self.templateFlux = templateFlux/templateFluxUnits
    self.templateWave = templateWave/templateWaveUnits

    self.templateResolution = templateDB['resolution']
    if self.templateResolution == 0:
      self.templateResolution = None

  @template.deleter
  def template(self):
    self._template = None
    try:
      del self.templateWave
      del self.templateFlux
      del self.templateResolution
    except AttributeError:
      pass
  ###

  #-- Initializing
  def copy(self):
    '''
      Method to copy this object by value into a new object. Returns the copy.

      Copys by value so affecting one does not affect the other.
    '''
    # initialize copy object
    theCopy = hrsObs(self.planet, self.instrument, self.date, self.order, self.template, self.dbPath)

    # List of kws to ignore in copying over
    ignore = ['dbPath', '_planet', '_instrument', '_date', '_order', '_template']

    for k,v in self.__dict__.items():
      if k in ignore:
        continue

      # Try and make value copies of each attribute in self
      # When unable (e.g. for a string attribute) calling .copy() is unneccessary
      try:
        setattr(theCopy, k, copy.deepcopy(v))
      except AttributeError:
        setattr(theCopy, k, v)

    return theCopy

  def keys(self):
    '''
      Returns all valid keys used for this object
    '''

    return list(self.__dict__.keys())

  def getDefaultSavePath(self, topDir='data/'):
    return topDir+f'{self._planet}/{self._date}/order_{self._order}/'

  def save(self, saveName, topDir='data/'):
    savePath = self.getDefaultSavePath(topDir)
    makePaths(savePath)
    fileName = savePath+saveName
    if fileName[-7:] != '.pickle':
      fileName = fileName + '.pickle'

    with open(fileName, 'wb') as f:
      pickle.dump(self, f)

  def load(self, saveName, topDir='data/', useGenericPath=True, path=False):
    if useGenericPath:
      path = self.getDefaultSavePath(topDir)

    fileName = path + saveName
    if fileName[-7:] != '.pickle':
      fileName = fileName + '.pickle'

    loaded = readFile(fileName)
    for key,value in loaded.__dict__.items():
      setattr(self, key, value)

  def initializeDatabase(self):
    '''
      Loads in values from database. Data loaded in is used to specify allowed values of attributes, and describes where to find actual data/ how to analyze it.
    '''

    # Check database type and existence
    dbType = self.dbPath.split('.')[-1]
    if dbType != 'json':
      raise TypeError('Currently only JSON databases are supported')

    try:
      with open(self.dbPath) as f:
        database = json.load(f)
    except (FileNotFoundError, UnicodeDecodeError, ValueError) as e:
      print('Database file must be pre-existing valid json file.')
      raise(e)

    self.database = database

  def updateDatabase(self, initialize=False):
    '''
      Descend into database to the level specified by object attributes If (attr) is specified, (new attr) is set as per:

      planet -> orbParams, starName, templates, templateFiles
      instrument -> observatory, dataFormat, dataPaths
      date -> defaultKWs
      order -> analysisKWs

      Parameters:
        initialize (bool) : re-initialize the database from the file
    '''
    # 
    if initialize:
      self.initializeDatabase()

    # Make sure database exists
    try:
      database = self.database
    except AttributeError:
      self.initializeDatabase()
      database = self.database

    self.planets = list(database.keys())

    # Determine planet level parameters
    if self.planet is not None:
      try:
        database = database[self.planet]
      except KeyError:
        raise KeyError("Planet "+self.planet+" not found.")
      self.database = database

      self.orbParams         = database['orbitalParamters']
      self.starName          = database['starName']
      self.starParams        = database['starParams']
      self.planetParams      = database['planetParams']
      self.templateDB        = database['templates']

      templatesList = list(database['templates'].keys())
      otherKeys = ['directory','wave_units', 'flux_units', 'resolution', 'log']
      templatesList = [t for t in templatesList if t not in otherKeys]
      self.templates = templatesList

      self.instruments = list(database['instruments'].keys())

      del self.planets
      
    # Set Instrument level parameters
    if self.instrument is not None:
      try:
        database = database['instruments'][self.instrument]
      except KeyError:
        raise KeyError('Instrument '+str(self.instrument)+' not found for planet "'+str(self.planet)+'".')
      self.database = database

      self.observatory = database['observatory']
      self.dataFormat = database['dataFormat']
      self.dataPaths  = database['dataPaths']
      self.dates      = list(database['dates'].keys())
      self.resolution = database['resolution']

      del self.instruments

    # Set Date level parameters
    if self.date is not None:
      try:
        database = database['dates'][self.date]
      except (KeyError, TypeError):
        raise KeyError('Date '+str(self.date)+' not found for planet "'+str(self.planet)+'", instrument "'+str(self.instrument)+'".')
      self.database = database

      try:
        self.dateLevelKeywords = database['dateLevelKeywords']
      except KeyError:
        pass

      del self.dates

    # Set Order level parameters
    if self.order is not None:
      try:
        self.orderLevelKeywords = database['orders'][str(self.order)]
      except KeyError:
        # This order is not in the database, print warning
        # warnings.warn('No order level keywords found for planet "'+str(self.planet)+'", instrument "'+str(self.instrument)+'", date "'+str(self.date)+'", order: '+str(self.order)+'.')
        self.orderLevelKeywords = {}

      del self.database

  def isValid(self, warn=True, fatal=False):
    '''
      Checks if this observation object is "valid"
      Valid means the dataset is fully specified -> 
      planet, instrument, date, order, template all determined

      if warn is true, prints highest level attribute which still needs to be specified
  
      Parameters:
        warn (bool) : If True, prints which needs to be specified 
        fatal (bool): raises error if not valid

      Returns:
        valid (bool): indicates if this dataset is valid
    '''
    attrList = [self.planet, self.instrument, self.date, self.order, self.template]
    attrNameList = ['planet','instrument','date','order','template']

    if warn or fatal:
      for i,attr in enumerate(attrList):
        if attr is None:
          if fatal:
            raise ValueError('Attribute "'+attrNameList[i]+'" is invalid.')
          else:
            print('Warning, attribute '+attrNameList[i]+' is invalid.')
          return False
      return True
    else:
      return not np.any([attr is None for attr in attrList])

  def collectRawData(self, verbose=True):
    '''
      Used to collect the raw data from disk for this observation
      Sets it as object Attributes

      Requires:
        isValid()
        self.dataFormat -> updateDataBase()
        self.dataPaths  -> updateDatabase()

      Creates:
        self.rawFlux
        self.error
        self.wavelengths
        self.times
        self.barycentricCorrection
    '''

    # check validity of observation 
    self.isValid(fatal=True)

    # check validity of dataFormat
    if self.dataFormat == 'order':
      dataDir    = self.dataPaths['dateDirectoryPrefix']+self.date+self.dataPaths['dateDirectorySuffix']
      dataFile   = dataDir + self.dataPaths['orderFilePrefix']+str(self.order)+self.dataPaths['orderFileSuffix']
      headerFile = dataDir+self.dataPaths['headerFile']

      rawData = readFile(dataFile)
      headers = readFile(headerFile)

      self.rawFlux = rawData['fluxes']
      self.error   = rawData['errors']
      self.wavelengths = rawData['waves']

      self.times = np.array(headers['JD'])
      self.airmass = np.array(headers['AIRMASS'])

    elif self.dataFormat == 'irtf':
      dataDir    = self.dataPaths['dateDirectoryPrefix']+self.date+self.dataPaths['dateDirectorySuffix']

      rawFlux = []
      wavelengths = []
      error = []
      times = []
      airmass = []

      # Load in data from files
      for fn in os.listdir(dataDir):
        prefixIndex = fn.find(self.dataPaths['filePrefix'])
        suffixIndex = fn.find(self.dataPaths['fileSuffix']) + len(self.dataPaths['fileSuffix'])

        # fn does not start with prefix/end with suffix
        if prefixIndex != 0 or suffixIndex != len(fn):
          continue

        data = fits.getdata(dataDir+fn)
        hdr  = fits.getheader(dataDir+fn)

        try:
          rawFlux.append(data[int(self.order), 1])
          wavelengths.append(data[int(self.order), 0])
          error.append(data[int(self.order), 2])
        except ValueError:
          raise ValueError('Order must be interpretable as an integer.')

        times.append(float(hdr['MJD'])+2400000.5)
        airmass.append(float(hdr['AM']))

      # Remove NaN's from data
      rawFlux = np.array(rawFlux)
      wavelengths = np.array(wavelengths)
      error = np.array(error)

      fnan = np.argwhere(np.isnan(rawFlux[0]))
      try:
        fnan = fnan[0,0]

        rawFlux = rawFlux[:,:fnan]
        wavelengths = wavelengths[:,:fnan]
        error = error[:,:fnan]
      except:
        pass

      self.rawFlux = rawFlux
      self.error = error
      self.wavelengths = wavelengths[0]
      self.times = np.array(times)
      self.airmass = np.array(airmass)

    else:
      raise ValueError('Currently only dataFormat "order" and "irtf" are accepted')

    # Try to load Barycentric Correction, write it if not found:
    barycentryVelocityFile = dataDir+'barycentricVelocity.pickle'
    try:
      self.barycentricCorrection = readFile(barycentryVelocityFile)
    except FileNotFoundError:
      # Barycentric Correction not on disk

      # Calculate it
      self.barycentricCorrection = getBarycentricCorrection(self.times, self.starName, self.observatory,verbose=verbose)

      # Write to disk
      with open(barycentryVelocityFile,'wb') as f:
        pickle.dump(self.barycentricCorrection,f)

    #set current data and keep track of order of operations
    self.data = self.rawFlux.copy()
    self.unProcessedData = self.rawFlux.copy()
    self.log = ['Raw Data']
  ###

  #-- Get features of data
  def getAlignmentOffset(self, padLen=None, peak_half_width=3,
                        upSampleFactor=1000, verbose=False
  ):
    '''
      Caclulates the alignment offset of each spectrum and the spectrum with the highest SNR. Returns that offset. 

      Meant to be called on raw/trimmed data

      Parameters:
        padLen (int): amount of zeros to pad to array before fft

        peak_half_width (int): number of points to include in a region around the xcor peak when upsampling

        upSampleFactor (int): factor by which to upsample the data when interpolating. Limits the precision 
            of the returned centers (i.e. an upSampleFactor of 10 can find peaks to a 0.1 precision level)

      Returns:
        offsets (1d-array): Wavelength offsets of each spectrum in pixel terms
    '''

    highSNR   = hru.getHighestSNR(self.data, self.error)

    offsets = hru.alignment(self.data, self.data[highSNR],
                        returnOffset=True, padLen=padLen,
                        peak_half_width=peak_half_width,
                        upSampleFactor=upSampleFactor,
                        verbose=verbose)

    return offsets

  def getRVs(self, unitRVs=False):
    '''
      Returns the RVs for this observation. Option to have unit RVs (i.e. set Kp=1, vsys=0)

      Parameters:
        unitRVs (bool): Whether or not RVs returned are unit.
      Returns:
        rvs (array): Array of radial velocity values
    '''
    orbParams = self.orbParams
    if unitRVs:
      orbParams = self.orbParams.copy()
      orbParams['Kp'] = 1
      orbParams['v_sys'] = 0

    rvs = getRV(self.times, **orbParams)

    return rvs

  def getObservedVelocities(self, kp=None, vsys=None, unitPrefix=1000):
    if kp is None:
      kp = self.getNominalKp()
    if vsys is None:
      vsys = self.getNominalVsys()

    return kp*self.getRVs(unitRVs=True) + self.barycentricCorrection/unitPrefix + vsys

  def getPhases(self):
    return getRV(self.times, **self.orbParams, returnPhase=True)

  def getTemplateInterp(self):
    interpolatedTemplate = interpolateData(self.templateFlux, self.templateWave, self.wavelengths, ext=2)
    return interpolatedTemplate

  def getNominalKp(self):
    return self.orbParams['Kp']

  def getNominalVsys(self):
    return self.orbParams['v_sys']

  def getLowResTemplate(self):
    templateMedW = np.median(self.templateWave)
    R_template = self.templateResolution
    if R_template is None:
      R_template = templateMedW/getSpacing(self.templateWave)
    obsMedW = np.median(self.wavelengths)

    lowResTemplate = reduceSpectralResolution(self.templateWave, self.templateFlux, self.resolution,
                                              R_template, obsMedW)

    return lowResTemplate

  def getStellarModel(self, blackbody=True):
    '''
    '''
    #TODO implement phoenix models
    if blackbody:
      return blackbodyFlux(self.wavelengths, self.starParams['teff']).value
    else:
      raise ValueError('Blackbody=False not yet implemented')
  ###

  #-- Processing Data
  '''
    These functions are for processing the raw data.
    All require raw data having been collected as per collectRawData()
  '''
  def trimData(self, manual=False,
               colTrimFunc=hru.findEdgeCuts_xcor,
               rowCuts='database', colCuts='database',
               figTitle=None, plotResult=False,
               neighborhood_size=30, gaussian_blur=10,
               edge=0, rightEdge=None, relative=True
  ):
    '''
      Runs highResUtils.trimData() on this observation. 
      Trims datasets according to rowCuts,colCuts and autoColTrimming

      Parameters:
        manual (bool): Whether or not to automatically trim columns

        plotResult (bool): shows trimming plots

        rowCuts (list of integers): Indicies of rows to remove from the data

        colCuts (list of integers): Indicies of columns to remove

        colTrimFunc (function): function to use to autoTrim cols. Requires manual=False.
                                Options (extra parameters):
                                  findEdgeCuts_xcor (neighborhood_size)
                                  findEdgeCuts_gradient (gaussian_blur, neighborhood_size)
                                  findEdgeCuts_numeric (edge, rightEdge, relative)

          colTrimFunc Parameters:
            findEdgeCuts_xcor:
              neighborhood_size (int): the region around each point to smooth snr/search for extrema

            findEdgeCuts_gradient:
              gaussian_blur (int) : the sigma to use when applying a gaussian filter to the data

              neighborhood_size (int): the region around each point to smooth snr/search for extrema

            findEdgeCuts_numeric:
              edge (int): how many points to trim from either side

              rightEdge (int): (optional) If entered, will use edge to trim from the left, rightEdge to trim from the right

              relative (bool): (optional) If true, cuts are relative to length of data (i.e. edge = 10 takes 10 points from each side). If false, cuts are absolute (i.e. edge = 10, rightEdge = 1100 cuts between 10 and 1100)
    '''

    # Set hard row and col cuts
    # keyword has priority over database
    # Allow keyword None to specify no cuts explicitly
    if rowCuts == 'database':
      try:
        rowCuts = self.dateLevelKeywords['rowCuts']
      except KeyError:
        rowCuts = None

    if colCuts == 'database':
      try:
        colCuts = self.dateLevelKeywords['colCuts']
      except KeyError:
        colCuts = None

    # run highResUtils.trimData()
    applyRowCuts  = [self.times, self.barycentricCorrection, self.airmass]
    applyColCuts  = [self.wavelengths]
    applyBothCuts = [self.error]

    if figTitle is None:
      figTitle='Date: '+self.date+', Order: '+str(self.order)

    data, applyRowCuts, applyColCuts, applyBothCuts = hru.trimData(self.data, 
                                          applyRowCuts, applyColCuts, applyBothCuts,
                                          rowCuts, colCuts, manual,
                                          colTrimFunc=colTrimFunc,
                                          plotResult=plotResult,
                                          figTitle=figTitle,
                                          neighborhood_size=neighborhood_size,
                                          gaussian_blur=gaussian_blur,
                                          edge=edge, rightEdge=rightEdge,
                                          relative=relative)

    # record results and log to order of operations
    self.data = data
    self.unProcessedData = data
    self.times                 = applyRowCuts[0]
    self.barycentricCorrection = applyRowCuts[1]
    self.airmass               = applyRowCuts[2]
    self.wavelengths           = applyColCuts[0]
    self.error                 = applyBothCuts[0]

    self.log.append('Trimmed')

  def alignData(self, iterations=1, ref=None, refNum=None,
                padLen=None, peak_half_width=3,
                upSampleFactor=1000, verbose=False
  ):
    '''
      Aligns the data to the wavelength solution of the spectrum with the highest SNR. 

      Parameters:
        iterations (int): Number of times to perform alignment 

        padLen (int): amount of zeros to pad to array before fft 

        peak_half_width (int): number of points to include in a region around the xcor peak when upsampling

        upSampleFactor (int): factor by which to upsample the data when interpolating. Limits the precision 
            of the returned centers (i.e. an upSampleFactor of 10 can find peaks to a 0.1 precision level)
    '''

    if ref is None:
      if refNum is None:
        highSNR = hru.getHighestSNR(self.data, self.error)
        ref = self.data[highSNR]
      else:
        ref = np.mean(self.data[:refNum],0)

    data, error = hru.alignment(self.data, ref,
                        iterations=iterations,
                        error=self.error, padLen=padLen,
                        peak_half_width=peak_half_width,
                        upSampleFactor=upSampleFactor,
                        verbose=verbose)

    self.data = data
    self.unProcessedData=data
    self.error = error

    self.log.append('Aligned')

  def removeLowFrequencyTrends(self, nTrends=1, kernel=65, mode=0, replaceMeans=True):
    '''
      Removes the first nTrends fourier components from each spectrum in the data.
      Does not remove the 0th component (mean).

      Parameters:
        nTrends (int): Number of fourier components to remove
    '''
    self.data = hru.removeLowFrequencyTrends(self.data, nTrends=nTrends,
      kernel=kernel, mode=mode, replaceMeans=replaceMeans)

    if mode == 0:
      self.log.append(str(nTrends)+" low freq trends removed")

    elif mode == 1:
      self.log.append(f'HighPass filter of width {kernel} applied')

  def normalizeData(self, normalizationScheme='divide_row', polyOrder=2):
    '''
      Apply a given normalizationScheme to the data.

      Options are:
        'subtract_col': subtract the mean column from each column
        'subtract_row': subtract the mean row (spectrum) from each spectra
        'subtract_all': apply both 'subtract_col' and 'subtract_row'
        'divide_col': divide data by mean column
        'divide_row': divide data by mean spectrum
        'divide_all': apply both 'divide_col' and 'divide_row'
        'polynomial': subtract a best fit polynomial of order 'polyOrder'
    '''
    data = hru.normalizeData(self.data, normalizationScheme=normalizationScheme, polyOrder=polyOrder)

    self.data = data
    self.log.append('Normalized: '+normalizationScheme)

  def generateMask(self, use_time_mask=True, use_wave_mask=False, plotResult=False,
                   relativeCutoff='default', absoluteCutoff='default', smoothingFactor=20, windowSize=25
  ):
    '''
      Generates a wavelength mask to apply to the data. Calls highResUtils.getTimeMask and highResUtils.getWaveMask to generate masks and combines them.

      Result saved as self.mask

      Parameters:
        use_time_mask (bool): Whether or not to generate a mask from hru.getTimeMask

        use_wave_mask (bool): Whether or not to generate a mask from hru.getWaveMask

        windowSize (int): Size of region around each wavelength column to consider for calculating SNR for waveMask

        relativeCutoff (positive float): Mask columns with SNR this sigma below the mean SNR

        absoluteCutoff (float): Mask columns with SNR below this value

        smoothingFactor (int): Number of columns around a masked column to also mask when combining

        plotResult(bool): plot each mask generated and the full mask
    '''

    # Try and read keywords from database
    if relativeCutoff == 'default':
      try:
        relativeCutoff = self.orderLevelKeywords['relativeCutoff']
      except KeyError:
        relativeCutoff = 3

    if absoluteCutoff == 'default':
      try:
        absoluteCutoff = self.orderLevelKeywords['absoluteCutoff']
      except KeyError:
        absoluteCutoff = 0

    time_mask = np.ones(np.shape(self.data)[1])
    wave_mask = np.ones(np.shape(self.data)[1])

    if use_time_mask:
      time_mask = hru.getTimeMask(self.data, relativeCutoff=relativeCutoff, absoluteCutoff=absoluteCutoff,
                                smoothingFactor=0, plotResult=plotResult)
    if use_wave_mask:
      wave_mask = hru.getWaveMask(self.data, windowSize=windowSize, relativeCutoff=relativeCutoff,
                                absoluteCutoff=absoluteCutoff, smoothingFactor=0,
                                plotResult=plotResult)

    mask = hru.combineMasks(time_mask, wave_mask, smoothingFactor=smoothingFactor)

    if plotResult:
      plt.figure()

      plt.plot(self.wavelengths, normalize(np.median(self.data,0)))
      plt.plot(mask)
      plt.ylim(-0.2,1.2)

      plt.title('Full Mask')
      plt.ylabel('Normalized Flux')
      plt.xlabel('Wavelength')

    self.log.append('Mask Created')

    self.mask = mask

  def applyMask(self):
    '''
      Applys mask to data.

      Mask should be created by calling generateMask(). If no mask has been created, will throw an error.
    '''
    try:
      data = hru.applyMask(self.data, self.mask)
    except AttributeError:
      raise AttributeError("A mask must be first created by calling generateMask() on this object.")

    self.data = data
    self.log.append('Masked')

  # todo 
  def sysrem(self, nCycles=None, verbose=False):
    '''
      Applies the Sysrem de-trending algorithm on this data.

      Parameters:
        nCycles (int): Number of times to run Sysrem

        verbose (bool): Print Sysrem progress updates
    '''

    # Try and read nCycles from database
    if nCycles is None:
      try:
        nCycles = self.orderLevelKeywords['sysremIterations']
      except KeyError:
        nCycles = 1

    data = hru.sysrem(self.data, self.error, nCycles=nCycles, verbose=verbose, returnAll=False)

    self.data = data
    self.sysremIterations = nCycles
    self.log.append('Sysrem: '+str(nCycles)+' cycles')

  def airmassFit(self, deg=2, log=False):
    '''
      As described in section 3.4 of Brogi+ 2016
    '''

    fittedData = hru.airmassFit(self.data, self.airmass, deg, log)

    if log:
      self.log.append(f'Divided through by airmass fit (to log data) of degree {deg}')
    else:
      self.log.append(f'Divided through by airmass fit of degree {deg}')

    self.data = fittedData

  def secondOrderAirmassFit(self, n_lines=3, manualLines=None, doPlot=False, title=''):
    '''
    '''

    corrected = hru.secondOrderAirmassFit(self.data, self.unProcessedData, n_lines, manualLines, doPlot, title=title)
    self.log.append(f'Applied second order airmass fit using {n_lines} columns.')
    self.data = corrected

  def getTellurics(self, neighborhood_size=20):
    rawSpec = np.median(self.unProcessedData,0)
    rawSpecMinima = getLocalMinima(rawSpec, neighborhood_size)

    

    residualVariance = np.var(self.data,0)
    residualVarianceMaxima = getLocalMaxima(residualVariance, neighborhood_size)

  def varianceWeight(self):
    '''
      Weights each column of data by its variance
    '''
    self.data = hru.varianceWeighting(self.data)
    self.log.append('Variance Weighted')
  ###

  #-- Comparing to Template
  def getFakePlanetSignal(self, injectedKp, injectedVsys, fudgeFactor=1,
    unitPrefix=1000, verbose=False
  ):
    lowResTemplate = self.getLowResTemplate()
    observedVelocities = self.getObservedVelocities(injectedKp, injectedVsys, unitPrefix)

    fakePlanetSignal = hru.generatePlanetSignal(lowResTemplate, self.templateWave, self.wavelengths,
      observedVelocities, unitPrefix, verbose=verbose)

    return fakePlanetSignal

  def injectFakeSignal(self, injectedKp, injectedVsys, fudgeFactor=1,
    Rp=None, unitPrefix=1000, verbose=False
  ):
    '''
      Injects the template signal into the data at the specified location and strength.

      Parameters:
        injectedKp (float): Kp of fake planet for injection.

        injectedVsys (float): Vsys of fake planet for injection.

        relativeStrength (float): Amplitude of template features relative to median of data

        unitPrefix (float): Units of velocity divided by meter/second. (Velocity units of injectedKp, injectedVsys)
        i.e. unitPrefix = 1000 implies velocity is in km/s
             unitPrefix = (1000 / 86400) implies velocity is km/day

        verbose (bool): If true, prints progress bar
    '''
    stellarModel = self.getStellarModel()
    if Rp is None:
      Rp = self.planetParams['radius'] #Jupiter radius

    Rs = self.starParams['radius'] * getUnitRatio(u.R_sun, u.R_jup) #Jupiter radius

    fakePlanetSignal = self.getFakePlanetSignal(injectedKp, injectedVsys, fudgeFactor,
      unitPrefix, verbose)

    fakeSignal = hru.generateFakeData(self.data, fakePlanetSignal, stellarModel, Rp/Rs,
      fudgeFactor=fudgeFactor, doInject=False)

    newData = self.data + fakeSignal
    self.log.append(f'Injected Fake Signal at {injectedKp}, {injectedVsys}, fudge::'+np.format_float_scientific(fudgeFactor))

    self.data = newData
    self.injection = {}
    self.injection['Kp'] = injectedKp
    self.injection['Vsys'] = injectedVsys
    self.injection['fudgeFactor'] = fudgeFactor

  def generateXCM(self, normalizeXCM=True, unitPrefix=1000, xcorMode='same', verbose=False):
    '''
      Calculates the cross-correlation matrix for this observation.

      Cross correlates the current stage of self.data with the template.

      Stores result as self.xcm,
      Stores alos self.crossCorVels

      Parameters:
        normalizeXCM (bool): whether or not to normalize the cross correlation functions according to Zucker 2003

        unitPrefix (float): Units of velocity divided by meter/second. (Velocity units of kpRange)
        i.e. unitPrefix = 1000 implies velocity is in km/s
             unitPrefix = (1000 / 86400) implies velocity is km/day

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
    '''
    # Interpolate template onto detector wavelengths.
    # If detector wavelength is not in template wavelengths, throw an error
    interpolatedTemplate = self.getTemplateInterp()

    self.xcm = hru.generateXCM(self.data, interpolatedTemplate, normalizeXCM=normalizeXCM,
                                xcorMode=xcorMode, verbose=verbose)
    self.unTrimmedXCM = self.xcm.copy()

    # Generate XCor velocities now to allow for plotting XCM, even though they'll be generated later too
    self.crossCorVels = hru.getCrossCorrelationVelocity(self.wavelengths, unitPrefix=unitPrefix)
    self.unTrimmedXCV = self.crossCorVels.copy()

    self.log.append('XCM Generated')

  def alignXCM(self, kp,
    unitPrefix=1000
  ):
    '''
    '''

    rvs = kp * self.getRVs(unitRVs=True)
    rvs = rvs + self.barycentricCorrection/unitPrefix

    alignedXCM = hru.alignXCM(self.xcm, self.crossCorVels, rvs, isInterpolatedXCM=False)

    return alignedXCM

  def generateSigMat(self, kpRange, unitPrefix=1000,
                     outputVelocities=None,
                     verbose=False):
    '''
      Generates a significance matrix for this observation. self.generateXCM() must be called before this function.

      Stores result as self.sigMat, and as self.unNormedSigMat
      Also stores kpRange which is needed to plot sigMat

      Parameters:
        kpRange (1d-array): List of Kp values to attempt in aligning to orbital solutions.

        unitPrefix (float): Units of velocity divided by meter/second. (Velocity units of kpRange)
        i.e. unitPrefix = 1000 implies velocity is in km/s
             unitPrefix = (1000 / 86400) implies velocity is km/day

        outputVelocities (array): velocities for sigMat to cover. Two Options:
                                  Length 2 array (e.g. [-100,100]):
                                    bounds velocities to this range but otherwise uses native resolution

                                  Length n array (e.g. np.arange(-100,100)):
                                    Interpolates results onto this velocity range. Useful for adding together several results

        verbose (bool): Whether or not to progressbar
    '''
    if outputVelocities is None:
      sigMat = hru.generateSigMat(self.xcm, kpRange, self.crossCorVels,
        self.getRVs(unitRVs=True), self.barycentricCorrection, unitPrefix=unitPrefix,
        verbose=verbose, xValsIsVelocity=True)
    else:
      sigMat, limitedVelocities, trimmedXCM = hru.generateSigMat(self.xcm, kpRange,
        self.crossCorVels, self.getRVs(unitRVs=True), self.barycentricCorrection,
        outputVelocities=outputVelocities, returnXcorVels=True,
        xValsIsVelocity=True, unitPrefix=unitPrefix, verbose=verbose)

      self.crossCorVels = limitedVelocities
      self.xcm = trimmedXCM

    self.kpRange = kpRange
    self.sigMat = sigMat
    self.unNormedSigMat = sigMat.copy()

    self.log.append('SigMat Generated')

  def reNormalizeSigMat(self, rowByRow=False, byPercentiles=False):
    '''
      Normalizes the significance matrix so each value represents a sigma rather than an arbitrary value.
      Divides the values in sigMat by the standard deviation of sigmat

      applies normalization to self.unNormedSigMat
      saves result as self.sigMat

      Parameters:
        rowByRow (bool): whether to normalize by the standard deviation of each row (rowByRow = True)
                         or the whole matrix (rowByRow = False)

        byPercentiles (bool): whether to normalize by the actual standard deviation (byPercentiles = False)
                              or by the 16th-84th percentiles (i.e. encompasing 68% of the data)
                                (byPercentiles=True)
    '''
    sigMat = hru.normalizeSigMat(self.unNormedSigMat, rowByRow=rowByRow, byPercentiles=byPercentiles)
    self.sigMat = sigMat
    logStr = f'SigMat Normalized, rowByRow: {rowByRow}, byPercentiles: {byPercentiles}'
    self.log.append(logStr)

  def reportDetectionStrength(self, targetKp=None, targetVsys=None,
                              kpSearchExtent=2, vsysSearchExtent=4,
                              unNormedSigMat=False, rowByRow=False,
                              byPercentiles=False,
                              plotResult=False, saveName=None,
                              plotKpExtent=40, plotVsysExtent=50,
                              clim=[None,None], title='',
                              figsize=None, cmap='viridis',
                              unitStr='km/s', show=True, close=False
  ):
    '''
      Reports the detection strength found within a region around targetKp, targetVsys

      Looks at a rectangular region centered on targetKp, targetVsys of width 2x vsysSearchExtent,
      height 2x kpSearchExtent. Finds the max value in this region and returns it as well as its coordinates

      If plotResult is true, sigMat is plotted with a container drawn around the search region
      the max value found in the search is marked with a triangle

      Parameters:
        targetKp (float): target Kp around which to search. Defaults to that in self.orbParams

        targetVsys (float): target Vsys around which to search. Defaults to that in self.orbParams

        kpSearchExtent(float): search will be performed encompassing region targetKp +- kpSearchExtent

        vsysSearchExtent(float): search will be performed encompassing region targetVsys +- vsysSearchExtent

        plotResult (bool): Plots sigmat if true

        saveName (str): If specified, will save figure at this location/name

        plotKpExtent (float): how far to go on either side of targetKp for plotting

        plotVsysExtent (float): how far to go on either side of targetVsys for plotting

        clim (length 2 array): Limits for colorbar

        title (str): Figure Title

        figsize (length 2 array): size of figure

        cmap (str): colormap to use

        unitStr (str): String to specify units on axes labels for both crossCorVels and kpRange

        show (bool): If true, calls plt.show()

        close(bool): If true, closes the plot

      Returns:
        detectionStrength (float): Maximum value of sigmat in search region

        detectionCoords (length 2 array): Coordinates to found point
    '''

    # Attempt to use self.orbParams if no target values are given
    if targetKp is None:
      try:
        targetKp = self.orbParams['Kp']
      except AttributeError:
        pass

    if targetVsys is None:
      try:
        targetVsys = self.orbParams['v_sys']
      except AttributeError:
        pass

    # Mysteriously, calling print here resolves a broken pipe error
    # caused by calling this function in a multiprocessing context.
    # Do not delete
    print('',end='')

    if unNormedSigMat:
      sm = self.unNormedSigMat
    else:
      sm = hru.normalizeSigMat(self.unNormedSigMat, rowByRow=rowByRow, byPercentiles=byPercentiles)

    detectionStrength, detectionCoords = hru.reportDetectionStrength(
                                            sm, self.crossCorVels, self.kpRange,
                                            targetKp, targetVsys, kpSearchExtent=kpSearchExtent,
                                            vsysSearchExtent=vsysSearchExtent, plotResult=plotResult,
                                            saveName=saveName, plotKpExtent=plotKpExtent,
                                            plotVsysExtent=plotVsysExtent, clim=clim, title=title,
                                            figsize=figsize, cmap=cmap, unitStr=unitStr, show=show)

    return detectionStrength, detectionCoords
  ###

  #-- Plotting
  def getNameString(self):
    nameStr = str(self.planet) + ' - '
    nameStr += str(self.instrument) + ' - '
    nameStr += str(self.date) + ' - '
    nameStr += 'Order: '+ str(self.order) + '\n'
    return nameStr

  def plotSigMat(self, xlim=[-100,100], ylim=None, clim=[None,None],
                  figsize=None, cmap='viridis', title='', saveName=None,
                  targetKp=None, targetVsys=None, unitStr='km/s', nDecimal=2
  ):
    '''
      Plots the significance matrix for this observation.
      Requires self.generateSigMat() has been called
    '''

    # Attempt to use self.orbParams if no target values are given
    if targetKp is None:
      try:
        targetKp = self.orbParams['Kp']
      except AttributeError:
        pass

    if targetVsys is None:
      try:
        targetVsys = self.orbParams['v_sys']
      except AttributeError:
        pass

    fullTitle = self.getNameString() + title

    hru.plotSigMat(self.sigMat, self.crossCorVels, self.kpRange,
                   targetKp=targetKp, targetVsys=targetVsys,
                   xlim=xlim, ylim=ylim, clim=clim,
                   figsize=figsize, cmap=cmap, title=fullTitle,
                   saveName=saveName, nDecimal=nDecimal)

  def plotData(self, yscale='frame', xlim=None, ylim=None,
               clim=[None,None], figsize=None, cmap='viridis',
               title='', saveName=None, wavelengthUnits='microns'
  ):
    '''
      Plots the current stage of self.data()
      Includes self.log in the title
    '''
    if yscale == 'frame':
      ys = np.arange(len(self.data))
      ylabel = 'Frame'
    elif yscale == 'time':
      ys = self.times
      ymin = np.min(ys)
      minDay = ymin - ymin%1
      ys = ys - minDay
      ylabel = 'Time (JD - '+str(minDay)+')'
    else:
      raise ValueError('yscale must be either "frame" or "time".')

    # Format the log to add to the title
    processStr = 'Data Process: '
    for i, entry in enumerate(self.log):
      if i%3 == 0:
        processStr+='\n'

      processStr += entry

      if i != len(self.log)-1:
        processStr += ' --> '

    if title != '' and title[-1] != '\n':
      title += '\n'

    fullTitle = self.getNameString() + title + processStr
    xlabel = 'Wavelength ('+str(wavelengthUnits)+')'

    hru.plotData(self.data, self.wavelengths, ys, xlabel, ylabel,
                 xlim=xlim, ylim=ylim, clim=clim,
                 figsize=figsize, cmap=cmap, title=fullTitle,
                 saveName=saveName)

  def plotXCM(self, yscale='frame', alignmentKp=None, unitPrefix=1000,
              xlim=None, ylim=None, clim=[None,None],
              figsize=None, cmap='viridis', title='',
              saveName=None, velocityUnits='km/s'
  ):
    '''
      Plots Cross Correlation matrix for this observation.
      If alignmentKp is specified, aligns the xcm to that value
    '''
    if yscale == 'frame':
      ys = np.arange(len(self.data))
      ylabel = 'Frame'
    elif yscale == 'time':
      ys = self.times
      ymin = np.min(ys)
      minDay = ymin - ymin%1
      ys = ys - minDay
      ylabel = 'Time (JD - '+str(minDay)+')'
    else:
      raise ValueError('yscale must be either "frame" or "time".')

    # Perform alignment steps
    xcm = self.xcm.copy()
    alignmentStr = ''
    if alignmentKp is not None:
      unitRVs = self.getRVs(unitRVs=True)

      rv = alignmentKp * unitRVs + self.barycentricCorrection/unitPrefix
      xcm = hru.alignXCM(xcm, self.crossCorVels, rv, isInterpolatedXCM=False)
      alignmentStr = 'Aligned to Kp='+str(alignmentKp)+' '+str(velocityUnits)+'.'

    # Get axes labels and plot
    if title != '' and title[-1]!='\n':
      title += '\n'

    fullTitle = self.getNameString() + title + alignmentStr
    xlabel = 'Cross Correlation Offset ('+str(velocityUnits)+')'

    hru.plotData(xcm, self.crossCorVels, ys, xlabel, ylabel,
                 xlim=xlim, ylim=ylim, clim=clim, figsize=figsize,
                 cmap=cmap, title=fullTitle, saveName=saveName)
  ###

  #-- Super-level functions
  def prepareData(self,
    # TrimData
    manual=False, plotTrim=False, colTrimFunc=hru.findEdgeCuts_xcor,
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
    doInjectSignal=False, removeNominal=False,
    injectedKp=None, injectedVsys=None,
    fudgeFactor=None, Rp=None, unitPrefix=1000,
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

    self.trimData(manual=manual, plotResult=plotTrim,
                  colTrimFunc=colTrimFunc,
                  neighborhood_size=neighborhood_size, gaussian_blur=gaussian_blur,
                  edge=edge, rightEdge=rightEdge, relative=relative)

    self.alignData(iterations=alignmentIterations, padLen=alignmentPadLen,
                   peak_half_width=alignmentPeakHalfWidth, upSampleFactor=alignmentUpSampFactor,
                   verbose=verbose)

    if doRemoveLFTrends:
      # After generate Mask?
      self.removeLowFrequencyTrends(nTrends=nTrends, kernel=hpKernel,
        mode=lfTrendMode)

    if doInjectSignal:
      # print('---------------------------------')
      # print('----- Injecting Fake Signal -----')
      # print('---------------------------------')

      if removeNominal:
        self.injectFakeSignal(injectedKp=self.getNominalKp(), injectedVsys=self.getNominalVsys(),
          fudgeFactor= -1, Rp=Rp, unitPrefix=unitPrefix)

      self.injectFakeSignal(injectedKp=injectedKp, injectedVsys=injectedVsys,
                            fudgeFactor=fudgeFactor, Rp=Rp,
                            unitPrefix=unitPrefix, verbose=verbose)

    self.generateMask(use_time_mask=use_time_mask, use_wave_mask=use_wave_mask, plotResult=plotMasks,
                      relativeCutoff=maskRelativeCutoff, absoluteCutoff=maskAbsoluteCutoff,
                      smoothingFactor=maskSmoothingFactor, windowSize=maskWindowSize)

    self.normalizeData(normalizationScheme=normalizationScheme, polyOrder=polyOrder)

    self.applyMask()

    if not stopBeforeSysrem:
      self.sysrem(nCycles=sysremIterations, verbose=verbose)

      if doVarianceWeight:
        self.varianceWeight()

      if highPassFilter:
        print(f'hpfiltering {hpKernel}')
        self.removeLowFrequencyTrends(mode=1, kernel=hpKernel, replaceMeans=False)

      self.applyMask()

  def xcorAnalysis(self, kpRange,
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
      Call self.prepareData() then self.xcorAnalysis()
    '''

    self.generateXCM(normalizeXCM=normalizeXCM, xcorMode=xcorMode,
                     unitPrefix=unitPrefix, verbose=verbose)

    self.generateSigMat(kpRange, unitPrefix=unitPrefix, outputVelocities=outputVelocities, verbose=verbose)

    if doNormalizeSigMat:
      self.reNormalizeSigMat(rowByRow=rowByRow, byPercentiles=byPercentiles)

  def prepareDataGeneric(self,
    refNum=None,
    normalizationScheme='divide_all',
    removeNominal=False,
    doInjectSignal=False,
    injectedKp=None, injectedVsys=None,
    injectionFudgeFactor=1, injectionRp=None,
    removeNominalStrength=None
  ):
    self.trimData()
    self.alignData(refNum=refNum)

    if removeNominal:
      strength = -1 * np.abs(removeNominalStrength)
      self.injectFakeSignal(self.getNominalKp(), self.getNominalVsys(), fudgeFactor=strength, Rp=injectionRp)

    if doInjectSignal:
      self.injectFakeSignal(injectedKp, injectedVsys,
        fudgeFactor=injectionFudgeFactor, Rp=injectionRp)

    self.generateMask()
    self.normalizeData(normalizationScheme)
    self.applyMask()

    return self
  
  def prepareDataAirmass(self,
    secondOrder=True,
    refNum=None,
    highPassFilter=False,
    normalizationScheme='divide_col',

    removeNominal=False,
    doInjectSignal=False,
    injectedKp=None, injectedVsys=None,
    injectionFudgeFactor=1, injectionRp=None,
    removeNominalStrength=None
  ):
    self.trimData()
    self.alignData(refNum=refNum)

    if removeNominalStrength is not None:
      strength = -1 * np.abs(removeNominalStrength)
      self.injectFakeSignal(self.getNominalKp(), self.getNominalVsys(), fudgeFactor=strength, Rp=injectionRp)

    if doInjectSignal:
      self.injectFakeSignal(injectedKp, injectedVsys,
        fudgeFactor=injectionFudgeFactor, Rp=injectionRp)

    self.generateMask()
    self.normalizeData(normalizationScheme)

    self.airmassFit()

    if highPassFilter:
      print('not implemented yet ya dingus')

    if secondOrder:
      self.secondOrderAirmassFit(doPlot=False)

    self.applyMask()
    self.varianceWeight()
    self.applyMask()

    return self

  ###