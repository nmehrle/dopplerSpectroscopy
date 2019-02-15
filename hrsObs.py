import numpy as np
import json

class hrsObs:
  '''
    An observation object

    Attributes:
      dbPath   (str)  : path to database
      database (dict) : dictionary holding onto all hardcoded information down to the level specified
  '''
  def __init__(self, dbPath, 
                planet=None, instrument=None, date=None, order=None,
                template=None):
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

    self.template = template

    self.updateDatabase(initialize=True)

  #-- Top Level Properties
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
    self._planet = None
    self.updateDatabase(initialize=True)
  
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
    self._instrument = None
    self.updateDatabase(initialize=True)

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
    self._date = None
    self.updateDatabase(initialize=True)
  
  @property
  def order(self):
    return self._order

  @order.setter
  def order(self,value):
    self._order = value
    self.updateDatabase(initialize=True)

  @order.deleter
  def order(self,value):
    self._order = None
    self.updateDatabase(initialize=True)
  ###

  def initializeDatabase(self):
    '''
      Loads in values from database.
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

      planet -> orbParams, starName, templates
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
      

    if self.instrument is not None:
      database = database['instruments'][self.instrument]
      self.database = database

      self.observatory = database['observatory']
      self.dataFormat = database['dataFormat']
      self.dataPaths  = database['dataPaths']
      self.dates      = list(database['dates'].keys())

      del self.instruments

    if self.date is not None:
      database = database['dates'][self.date]
      self.database = database

      del self.dates






