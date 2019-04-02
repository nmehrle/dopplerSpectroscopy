import json

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
      templateUnits = self.allTemplateUnits[value]
    except:
      raise KeyError('Units not specified for template "' + str(value) + '". Please enter the value used to convert to microns into the database, i.e. if the template data is in angstroms, enter 10000.')

    try:
      templateData = readFile(templateFile)
    except ValueError:
      raise ValueError('Problem with database entry: template "'+str(value)+'", file "'+str(templateFile)+'" not supported.')
    except FileNotFoundError:
      raise FileNotFoundError('Template File "'+str(templateFile)+'" not found.')

    self._template = value
    self.templateWave = templateData['wavelengths']/templateUnits
    self.templateFlux = templateData['flux']

  @template.deleter
  def template(self):
    self._template = None
    try:
      del self.templateWave
      del self.templateFlux
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
    theCopy = hrsObs(self.dbPath, self.planet, self.instrument, self.date, self.order, self.template)

    # List of kws to ignore in copying over
    ignore = ['dbPath', '_planet', '_instrument', '_date', '_order', '_template']

    for k,v in self.__dict__.items():
      if k in ignore:
        continue

      # Try and make value copies of each attribute in self
      # When unable (e.g. for a string attribute) calling .copy() is unneccessary
      try:
        setattr(theCopy, k, v.copy())
      except AttributeError:
        setattr(theCopy, k, v)

    return theCopy

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
      self.templateFiles     = database['templates']
      self.allTemplateUnits  = database['templateUnits']

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

    # Set Order level parameters
    if self.order is not None:
      try:
        self.orderLevelKeywords = database['orders'][str(self.order)]
      except KeyError:
        # This order is not in the database, print warning
        print('Warning, No order level keywords found for planet "'+str(self.planet)+'", instrument "'+str(self.instrument)+'", date "'+str(self.date)+'", order: '+str(self.order)+'.')
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

    elif self.dataFormat == 'irtf':
      dataDir    = self.dataPaths['dateDirectoryPrefix']+self.date+self.dataPaths['dateDirectorySuffix']

      rawFlux = []
      wavelengths = []
      error = []
      times = []

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
  ###

  #-- Processing Data
  '''
    These functions are for processing the raw data.
    All require raw data having been collected as per collectRawData()
  '''
  def trimData(self, doAutoTrimCols=True,
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
        doAutoTrimCols (bool): Whether or not to automatically trim columns

        plotResult (bool): shows trimming plots

        rowCuts (list of integers): Indicies of rows to remove from the data

        colCuts (list of integers): Indicies of columns to remove

        colTrimFunc (function): function to use to autoTrim cols. Requires doAutoTrimCols=True.
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
    applyRowCuts  = [self.times, self.barycentricCorrection]
    applyColCuts  = [self.wavelengths]
    applyBothCuts = [self.error]

    if figTitle is None:
      figTitle='Date: '+self.date+', Order: '+str(self.order)

    data, applyRowCuts, applyColCuts, applyBothCuts = hru.trimData(self.data, 
                                          applyRowCuts, applyColCuts, applyBothCuts,
                                          rowCuts, colCuts, doAutoTrimCols,
                                          colTrimFunc=colTrimFunc,
                                          plotResult=plotResult,
                                          figTitle=figTitle,
                                          neighborhood_size=neighborhood_size,
                                          gaussian_blur=gaussian_blur,
                                          edge=edge, rightEdge=rightEdge,
                                          relative=relative)

    # record results and log to order of operations
    self.data = data
    self.times                 = applyRowCuts[0]
    self.barycentricCorrection = applyRowCuts[1]
    self.wavelengths           = applyColCuts[0]
    self.error                 = applyBothCuts[0]

    self.log.append('Trimmed')

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

  def removeLowFrequencyTrends(self, nTrends=1):
    '''
      Removes the first nTrends fourier components from each spectrum in the data.
      Does not remove the 0th component (mean).

      Parameters:
        nTrends (int): Number of fourier components to remove
    '''
    self.data = hru.removeLowFrequencyTrends(self.data, nTrends=nTrends)
    self.log.append(str(nTrends)+" low freq trends removed")

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
    self.log.append('Sysrem: '+str(nCycles)+' cycles')

  def varianceWeight(self):
    '''
      Weights each column of data by its variance
    '''
    self.data = hru.varianceWeighting(self.data)
    self.log.append('Variance Weighted')
  ###

  #-- Comparing to Template
  def injectFakeSignal(self, injectedKp, injectedVsys, relativeStrength, unitPrefix=1000, verbose=False):
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
    fakeSignal = hru.generateFakeSignal(self.data, self.wavelengths, self.getRVs(unitRVs=True),
                                        self.barycentricCorrection, injectedKp, injectedVsys, self.templateFlux,
                                        self.templateWave, relativeStrength=relativeStrength,
                                        unitPrefix=unitPrefix, verbose=verbose, returnInjection=True)

    self.data = fakeSignal
    self.log.append('Injected Fake Signal at '+np.format_float_scientific(relativeStrength))

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
    interpolatedTemplate = interpolateData(self.templateFlux, self.templateWave, self.wavelengths, ext=2)

    self.xcm = hru.generateXCM(self.data, interpolatedTemplate, normalizeXCM=normalizeXCM,
                                xcorMode=xcorMode, verbose=verbose)

    # Generate XCor velocities now to allow for plotting XCM, even though they'll be generated later too
    self.crossCorVels = hru.getCrossCorrelationVelocity(self.wavelengths, unitPrefix=unitPrefix)

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
      sigMat = hru.generateSigMat(self.xcm, kpRange, self.wavelengths, self.getRVs(unitRVs=True),
                                self.barycentricCorrection, unitPrefix=unitPrefix, verbose=verbose)
    else:
      sigMat, limitedVelocities = hru.generateSigMat(self.xcm, kpRange, self.wavelengths,
                                self.getRVs(unitRVs=True), self.barycentricCorrection,
                                outputVelocities=outputVelocities, returnXcorVels=True,
                                unitPrefix=unitPrefix, verbose=verbose)
      self.crossCorVels = limitedVelocities

    self.kpRange = kpRange
    self.sigMat = sigMat
    self.unNormedSigMat = sigMat.copy()

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

  def reportDetectionStrength(self, targetKp=None, targetVsys=None,
                              kpSearchExtent=2, vsysSearchExtent=4,
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
    detectionStrength, detectionCoords = hru.reportDetectionStrength(
                                            self.sigMat, self.crossCorVels, self.kpRange,
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
                  targetKp=None, targetVsys=None, unitStr='km/s'
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
                   saveName=saveName)

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