import composite
import numpy as np

import warnings
warnings.simplefilter('ignore')

import messenger
from scipy import signal

kpr = np.arange(101)

planet='ups-and'
template='inverted'

aries='aries'
ariesDates=['2016oct16','2016oct17']
ariesOrders=np.arange(8)

ishell = 'ishell'
ishellDates = '2016nov04'
ishellOrders=np.arange(11,24)

def ishell1(obs, doInjectSignal=False,
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
  
  obs.removeLowFrequencyTrends(mode=0)
  obs.generateMask()
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

def ishell5(obs, doInjectSignal=False,
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
  obs.removeLowFrequencyTrends(mode=0)
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

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

def aries2(obs, doInjectSignal=False,
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
  
  obs.removeLowFrequencyTrends(mode=0)
  obs.generateMask()
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

def aries3(obs, doInjectSignal=False,
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
  
  obs.removeLowFrequencyTrends(mode=1)
  obs.generateMask()
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

def aries4(obs, doInjectSignal=False,
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
  obs.removeLowFrequencyTrends(mode=0)
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs

def aries5(obs, doInjectSignal=False,
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
  obs.removeLowFrequencyTrends(mode=1)
  obs.normalizeData(normalizationScheme)
  obs.applyMask()
  
  return obs


folder = 'plots/upsand/oldDefaults/'
cores = 4
doInjectSignal = False
injectionStrengths = None

sendMessage=True

targetVsysList=[1]

ariesFuncs = [aries2, aries3, aries4, aries5]
# ariesFuncs = [aries1]
ishellFuncs=[]
# ishellFuncs = [ishell1, ishell5, ishell7]


try:
  composite.generateSysremLandscape(planet, template,
    aries, ariesDates, ariesOrders,
    kpr, folder, cores=cores,
    doInjectSignal=False)
except Exception as e:
  messenger.sms(f'Failed on Aries! {e}')
  raise(e)

try:
  composite.generateSysremLandscape(planet, template,
    ishell, ishellDates, ishellOrders,
    kpr, folder, cores=cores,
    doInjectSignal=False)
except Exception as e:
  messenger.sms(f'Failed on ishell! {e}')
  raise(e)

messenger.sms('oldDefaults is Finished')

# for tvs in targetVsysList:
#   for highPassFilter in [False,True]:
#     if highPassFilter:
#       hpfol = folder+'hpfilt/'
#     else:
#       hpfol = folder
#     for normalizationScheme in ['divide_row','divide_all','divide_col']:
#       nsfol = hpfol + normalizationScheme+'/'

#       for af in ariesFuncs:
#         fol = nsfol+str(af.__name__)+'/'
#         print(fol)
#         try:
#           composite.generateSysremLandscape(planet, template,
#                             aries, ariesDates, ariesOrders,
#                             kpr, fol,
#                             cores=cores,
#                             prepareFunction=af,
#                             doInjectSignal=doInjectSignal,
#                             injectionStrengths=injectionStrengths,
#                             # targetKp=55, targetVsys=tvs,
#                             normalizationScheme=normalizationScheme,
#                             highPassFilter=highPassFilter)
#         except Exception as e:
#           messenger.sms('Failed! '+str(e))
#           raise(e)

#         if sendMessage:
#           # messenger.sms('running well')
#           sendMessage=False

#       for isf in ishellFuncs:
#         fol = nsfol+str(isf.__name__)+'/'
#         print(fol)
#         try:
#           composite.generateSysremLandscape(planet, template,
#                             ishell, ishellDates, ishellOrders,
#                             kpr, fol,
#                             cores=cores,
#                             prepareFunction=isf,
#                             doInjectSignal=doInjectSignal,
#                             injectionStrengths=injectionStrengths,
#                             # targetKp=55, targetVsys=tvs,
#                             normalizationScheme=normalizationScheme,
#                             highPassFilter=highPassFilter)
#         except Exception as e:
#           messenger.sms('Failed! '+str(e))
#           raise(e)

messenger.sms('Finished')