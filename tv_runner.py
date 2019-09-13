import composite
import numpy as np

import warnings
warnings.simplefilter('ignore')

import messenger
from scipy import signal


planet='ups-and'
templates = ['inverted']#, 'non-inverted']

instruments = ['aries','ishell']
dates = [['2016oct16','2016oct17'],['2016nov04']]
orders = [np.arange(8),np.arange(11,24)]

aries='aries'
ariesDates=['2016oct16','2016oct17']
ariesOrders=np.arange(8)

ishell = 'ishell'
ishellDates = '2016nov04'
ishellOrders=np.arange(11,24)

kpr = np.arange(101)


def ishell7(obs, doInjectSignal=False,
  injectedRelativeStrength=1,
  subtractSignal=False,
  injectedKp=55, injectedVsys=-30,
  normalizationScheme='divide_row'
):
  obs.trimData()
  obs.alignData()
  
  if doInjectSignal:
    obs.injectFakeSignal(injectedKp, injectedVsys, injectedRelativeStrength,
                         subtract=subtractSignal)

  obs.generateMask()
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

def aries1(obs, doInjectSignal=False,
  injectedRelativeStrength=1,
  subtractSignal=False,
  injectedKp=55, injectedVsys=-30,
  normalizationScheme='divide_row'
):
  obs.trimData()
  obs.alignData()
  
  if doInjectSignal:
    obs.injectFakeSignal(injectedKp, injectedVsys, injectedRelativeStrength,
                        subtract=subtractSignal)
  
  obs.generateMask()
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs


folder = 'plots/upsand/ishell7/'
cores = 4
doInjectSignal = False
injectionStrengths = None

try:
  composite.generateSysremLandscape(planet, templates,
    aries, ariesDates, ariesOrders,
    kpr, folder, cores=cores,
    doInjectSignal=doInjectSignal,
    injectionStrengths=injectionStrengths,
    prepareFunction=aries1,
    normalizationScheme='divide_all',
    highPassFilter=False
    )
except Exception as e:
  messenger.sms(f'Failed on Aries! {e}')
  raise(e)

try:
  composite.generateSysremLandscape(planet, templates,
    ishell, ishellDates, ishellOrders,
    kpr, folder, cores=cores,
    doInjectSignal=doInjectSignal,
    injectionStrengths=injectionStrengths,
    prepareFunction=ishell7,
    normalizationScheme='divide_all',
    highPassFilter=True
    )
except Exception as e:
  messenger.sms(f'Failed on ishell! {e}')
  raise(e)

messenger.sms('Done generating Data')

try:
  falsePositiveTest(planet, templates, instruments, dates, orders,
    folder, kpr, np.arange(-150,150), messageAtEnd=True, verbose=True)
except Exception as e:
  messenger.sms('Failed! '+str(e))
  raise(e)