
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto Final (Analisis Técnico)                                                          -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: Equipo 3 MICROESTRUCTURA Y SISTEMAS DE TRADING PRIM. 2023                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/if723286/MyST_LAB5_TEAM3                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

"""
# This function was used to initially download
# all the prices from mt5 to then be used
# in future labs and different computers
# To run again remove comment and import MetaTrader as mt5.
"""
 
def f_import_mt5(list_tickers):
    '''
    Imports prices from MetaTrader5
    given a list of tickers.
    Parameters
    ---------
    list_tickers : list
        list of tickers
    Returns
    --------
    mt5_rates : dict
        dictionary of DataFrames
        with prices of tickers
    '''
    mt5.initialize()
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2023,5,1,tzinfo=timezone)
    mt5_rates = {}
    for i in list_tickers:
        rates=(mt5.copy_rates_from(i,mt5.TIMEFRAME_M15,utc_from,60000))
        rates=pd.DataFrame(rates)
        rates['time'] = pd.to_datetime(rates['time'], unit='s')
        #rates.to_csv('')
        mt5_rates[i]=rates
        mt5_rates[i].to_csv('files/'+i+'.csv')
    mt5.shutdown()
    return mt5_rates
