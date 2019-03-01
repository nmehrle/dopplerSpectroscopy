import json

from utility import *
import highResUtils as hru

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi


class hrsObs:
  '''
    An observation object

    Attributes:
      dbPath   (str)  : path to database
      database (dict) : dictionary holding onto all hardcoded information down to the level specified
  '''
  def __init__(self, dbPath, 
                planet=None, instrument=None, date=None, order=None,
                template='default'
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
      

    try:
      templateFile = self.templateFiles['directory'] + self.templateFiles[value]
    except AttributeError:
      if self.planet is None:
        raise AttributeError('Must specify planet before template.')
      else:
        raise AttributeError('templateFiles not found for planet "'+str(self.planet)+'". Please check validity of database.')
    except KeyError:
      raise KeyError('Template "'+str(value)+'" not defined for planet "'+str(self.planet)+'".')

    try:
      self.templateData = readFile(templateFile)
    except ValueError:
      raise ValueError('Problem with database entry: template "'+str(value)+'", file "'+str(templateFile)+'" not supported.')
    except FileNotFoundError:
      raise FileNotFoundError('Template File "'+str(templateFile)+'" not found.')

    self._template = value

  @template.deleter
  def template(self):
    self._template = None
    try:
      del self.templateData
    except AttributeError:
      pass
  ###

  #-- Initializing 
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
      Descend into databse to the level specified by object attributes If (attr) is specified, (new attr) is set as per:

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
      self.templateFiles     = database['templates']
      self.templates         = list(database['templates'].keys())
      self.templates.remove('directory')

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

      del self.instruments

    # Set Date level parameters
    if self.date is not None:
      try:
        database = database['dates'][self.date]
      except KeyError:
        raise KeyError('Date '+str(self.date)+' not found for planet "'+str(self.planet)+'", instrument "'+str(self.instrument)+'".')
      self.database = database

      try:
        self.dateLevelKeywords = database['dateLevelKeywords']
      except KeyError:
        pass

      del self.dates

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

    else:
      raise ValueError('Currently only dataFormat "order" is accepted')

    #set current data and keep track of order of operations
    self.data = self.rawFlux.copy()
    self.log = ['Raw Data']
  ###

  #-- Processing Data
  '''
    These functions are for processing the raw data.
    All require raw data having been collected as per collectRawData()
  '''

  def trimData(self, doAutoTrimCols = False, plotResult=False, **kwargs):
    '''
      Runs highResUtils.trimData() on this observation. 
      Trims datasets according to rowCuts,colCuts and autoColTrimming

      Parameters:
        doAutoTrimCols (bool): Whether or not to automatically trim columns

        plotResult (bool): shows trimming plots

        **kwargs:
          entries should be:
            rowCuts (list of integers): Indicies of rows to remove from the data

            colCuts (list of integers): Indicies of columns to remove

            colTrimFunc (function): function to use to autoTrim cols. Requires doAutoTrimCols=True.
                                Options (extra parameters):
                                  hru.findEdgeCuts_xcor (neighborhood_size)
                                  hru.findEdgeCuts_gradient (gaussian_blur, neighborhood_size)
                                  hru.findEdgeCuts_numeric (edge, rightEdge, relative)
    '''

    # Set hard row and col cuts
    # keyword has priority over database
    try:
      rowCuts = kwargs.pop('rowCuts')
    except KeyError:
      try:
        rowCuts = self.dateLevelKeywords['rowCuts']
      except KeyError:
        rowCuts = None

    try:
      colCuts = kwargs.pop('colCuts')
    except KeyError:
      try:
        colCuts = self.dateLevelKeywords['colCuts']
      except KeyError:
        colCuts = None

    # run highResUtils.trimData()
    applyRowCuts  = [self.times, self.barycentricCorrection]
    applyColCuts  = [self.wavelengths]
    applyBothCuts = [self.error]

    try:
      figTitle = kwargs.pop('figTile')
    except KeyError:
      figTile='Date: '+self.date+', Order: '+str(self.order) 

    data, applyRowCuts, applyColCuts, applyBothCuts = hru.trimData(self.data, 
                                          applyRowCuts, applyColCuts, applyBothCuts,
                                          rowCuts, colCuts, doAutoTrimCols,
                                          plotResult=plotResult,
                                          figTile=figTile,
                                          **kwargs) 

    # record results and log to order of operations
    self.data = data
    self.times                 = applyRowCuts[0]
    self.barycentricCorrection = applyRowCuts[1]
    self.wavelengths           = applyColCuts[0]
    self.error                 = applyBothCuts[0]

    self.log.append('Trimmed')

  def getAlignmentOffset(self, padLen=None, peak_half_width=3,
                        upSampleFactor=1000, verbose=False
  ):
    '''
      Caclulates the alignment offset of each spectrum and the spectrum with the highest SNR. Returns that offset. 

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

  def alignData(self, iterations=1, padLen=None,
                peak_half_width=3, upSampleFactor=1000,
                verbose=False
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

    highSNR   = hru.getHighestSNR(self.data, self.error)

    data, error = hru.alignment(self.data, self.data[highSNR],
                        iterations=iterations,
                        error=self.error, padLen=padLen,
                        peak_half_width=peak_half_width,
                        upSampleFactor=upSampleFactor,
                        verbose=verbose)

    self.data = data
    self.error = error

    self.log.append('Aligned')

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
    data = self.data.copy()

    if normalizationScheme == 'subtract_col':
      data = data-np.mean(data,1)[:,np.newaxis]
    elif normalizationScheme == 'subtract_row':
      data = data-np.mean(data,0)
    elif normalizationScheme == 'subtract_all':
      data = data-np.mean(data,0)
      data = data-np.mean(data,1)[:,np.newaxis]
    elif normalizationScheme == 'divide_col':
      data = data / np.mean(data,1)[:,np.newaxis]
    elif normalizationScheme == 'divide_row':
      data = data / np.mean(data,0)
    elif normalizationScheme == 'divide_all':
      data = data / (np.mean(data,0) * np.mean(data,1)[:,np.newaxis])
    elif normalizationScheme == 'continuum':
      data = polynomialSubtract(data, polyOrder)
    else:
      raise(KeyError('Normalization Keyword '+normalizationScheme+' invalid. Valid KWs are a combination of "subtract, divide" and "row, col, all" e.g. "subtract_row". Or "continuum", with a valid Continuum Order'))

    self.data = data
    self.log.append('Normalized: '+normalizationScheme)

  def generateMask(self, use_time_mask=True, use_wave_mask=False, plotResult=False,
                   relativeCutoff=3, absoluteCutoff=0, smoothingFactor=20, windowSize=25
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
    time_mask = np.ones(np.shape(self.data)[1])
    wave_mask = np.ones(np.shape(self.data)[1])

    if use_time_mask:
      time_mask = hru.getTimeMask(self.data, relativeCutoff=relativeCutoff, absoluteCutoff=absoluteCutoff,
                                smoothingFactor=0, plotResult=plotResult)
    if use_wave_mask:
      wave_mask = hru.getWaveMask(self.data, windowSize=windowSize, relativeCutoff=relativeCutoff,
                                absoluteCutoff=absoluteCutoff, smoothingFactor=0,
                                plotResult=plotResult)

    mask = combineMasks(time_mask, wave_mask, smoothingFactor=smoothingFactor)

    if plotResult:
      plt.figure()

      plt.plot(self.wavelengths, normalize(np.median(self.data,0)))
      plt.plot(mask)
      plt.ylim(-0.2,1.2)

      plt.title('Full Mask')
      plt.ylabel('Normalized Flux')
      plt.xlabel('Wavelength')

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

  def sysrem(self, nCycles=1, verbose=False):
    '''
      Applies the Sysrem de-trending algorithm on this data.

      Parameters:
        nCycles (int): Number of times to run Sysrem

        verbose (bool): Print Sysrem progress updates
    '''

    data = sysrem(self.data, self.error, verbose=verbose, returnAll=False)

    self.data = data
    self.log.append('Sysrem: '+str(nCycles)+' cycles')

  def varianceWeight(self):
    '''
      Weights each column of data by its variance
    '''

    self.data = hru.varianceWeighting(self.data)
    self.log.append('Variance Weigthed')
  ###