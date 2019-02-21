import json

from utility import *

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
        self.errors
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
      self.errors  = rawData['errors']
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
  ###

  #-- Processing Data
  '''
    These functions are for processing the raw data.
    All require raw data having been collected as per collectRawData()
  '''

  def trimData():
    '''

    '''
    return 1

  def findEdgeCuts_xcor(flux, neighborhood_size=30, plotResult=False):
    '''
      Detects 'edges' in flux as regions where time-wise SNR dramatically drops off from the center. 

      Find edges by cross correlating snr with a step function and getting first minima/last maxima. 

      Parameters:
        flux (array): 2d array of flux as (time, wavelength)

        neighborhood_size (int): the region around each point to smooth snr/search for extrema

        plotResult (bool): if True, plot the cuts to visualize

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
      plt.figure()
      norm_snr = normalize(col_snr)
      norm_smooth = normalize(smooth)

      plt.plot(norm_snr-np.median(norm_snr),label='Column SNR')
      plt.plot(norm_smooth - np.median(norm_smooth),label='Minimum Filter')
      plt.plot(normalize(xcor, (-0.5,0)),label='Cross Correlation')
      plt.plot(normalize(np.median(flux,0),(-1,-0.5)), label='Median Flux')
      plt.plot((left_bound,left_bound),(-1.0,0), color='C2')
      plt.plot((right_bound,right_bound),(-1.0,0), color='C2')
      plt.legend()

      plt.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
      plt.set_xlabel('Column Number')
      plt.set_ylabel('Normalized SNR')
      plt.show()

    return left_bound, right_bound

  def findEdgeCuts_gradient(flux, gaussian_blur = 10, neighborhood_size = 30,
                            plotResult=False
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

    rightDelta = rightEdge - minima2
    rightCorner = rightEdge - np.min(rightDelta[rightDelta>0])

    leftDelta  = minima2 - leftEdge
    leftCorner = np.min(leftDelta[leftDelta>0]) + leftEdge

    if plotResult:
      plt.figure()
      norm_data = normalize(signal)
      norm_smooth = normalize(smooth)

      plt.plot(norm_data-np.median(norm_data),label='Data')
      plt.plot(norm_smooth - np.median(norm_smooth),label='Smoothed')
      plt.plot((leftCorner,leftCorner),(-0.5,0.5), color='C3')
      plt.plot((rightCorner,rightCorner),(-0.5,0.5), color='C3')

      plt.legend()
      plt.set_title('Edge Trimming\nLeft: '+str(leftCorner)+', Right: '+str(rightCorner))
      plt.set_xlabel('Column Number')
      plt.set_ylabel('Normalized Flux')

      plt.show()

    return leftCorner, rightCorner

  def findEdgeCuts_numeric(flux, edge, rightEdge=None, relative=True, plotResult=False):
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
      normData = normalize(np.median(flux,0))
      plt.figure()
      plt.plot(normData-np.median(normData),label='Median Spectrum')
      plt.plot((left_bound,left_bound),(-0.5,0.5), color='C2')
      plt.plot((right_bound,right_bound),(-0.5,0.5), color='C2')
      plt.legend()

      plt.set_title('Edge Trimming\nLeft: '+str(left_bound)+', Right: '+str(right_bound))
      plt.set_xlabel('Column Number')
      plt.set_ylabel('Normalized Flux')
      plt.show()

    return left_bound, right_bound

 
  ###

